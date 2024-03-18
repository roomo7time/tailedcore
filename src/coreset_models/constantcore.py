

class ConstantCore(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.1,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
        max_coreset_size=None,
    ):
        super(ConstantCore, self).__init__()

        self.feature_embedder = feature_embedder

        if max_coreset_size is not None:
            max_coreset_shape = (
                max_coreset_size,
                feature_embedder.get_feature_map_shape()[-1],
            )
        else:
            max_coreset_shape = None

        self.sampler = IncrementalGreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
            max_coreset_shape=max_coreset_shape,
        )

        nn_method = FaissNN(faiss_on_gpu, faiss_num_workers, device=device.index)
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.feature_map_shape = self.feature_embedder.get_feature_map_shape()

        self.rescale_segmentor = RescaleSegmentor(device=device, target_size=imagesize)

        if save_dir_path:
            self.coreset_path = os.path.join(save_dir_path, "coreset.pt")
        else:
            self.coreset_path = None

    def fit(self, trainloader: DataLoader, set_predictor=True):

        tqdm.write("Fitting...")

        coreset_features = self._load_coreset_features()
        if coreset_features is None:
            coreset_features = self._get_coreset(trainloader)
            self._save_coreset_features(coreset_features)

        if set_predictor:
            self.anomaly_scorer.fit([coreset_features])

    def incremental_fit(self, trainloader, set_predictor=True, save_coreset=True):

        if hasattr(self, "_base_coreset"):
            base_coreset = self._base_coreset
        else:
            self._backup_coreset()
            base_coreset = self._load_coreset_features()
            self._base_coreset = base_coreset

        coreset_features = self._get_coreset(trainloader, base_coreset=base_coreset)

        self._base_coreset = coreset_features

        if save_coreset:
            self._save_coreset_features(coreset_features)

        if set_predictor:
            self.anomaly_scorer.fit([coreset_features])

    def _backup_coreset(self):
        backup_coreset_path = os.path.join(
            os.path.dirname(self.coreset_path), "coreset_backup.npy"
        )
        if os.path.exists(backup_coreset_path):
            return
        shutil.copyfile(self.coreset_path, backup_coreset_path)

    def _load_coreset_features(self) -> torch.Tensor:
        if self.coreset_path and os.path.exists(self.coreset_path):
            coreset_features = torch.load(self.coreset_path)
            tqdm.write("Loaded a saved coreset!")
            return coreset_features
        else:
            return None

    def _save_coreset_features(self, coreset_features: torch.Tensor):
        if self.coreset_path:
            tqdm.write(f"Saving a coreset at {self.coreset_path}")
            os.makedirs(os.path.dirname(self.coreset_path), exist_ok=True)
            torch.save(coreset_features.clone().cpu().detach(), self.coreset_path)
            tqdm.write("Saved a coreset!")

    def predict_on(self, testloader: DataLoader):
        features = self._get_features(testloader)
        image_scores, score_masks = self._get_scores(features)

        return image_scores, score_masks

    def _get_scores(self, features: np.ndarray) -> np.ndarray:
        batch_size = features.shape[0] // (
            self.feature_map_shape[0] * self.feature_map_shape[1]
        )
        _scores, _, _indices = self.anomaly_scorer.predict([features])

        scores = torch.from_numpy(_scores)

        image_scores = torch.max(scores.reshape(batch_size, -1), dim=-1).values
        patch_scores = scores.reshape(
            batch_size, self.feature_map_shape[0], self.feature_map_shape[1]
        )

        score_masks = self.rescale_segmentor.convert_to_segmentation(patch_scores)

        return image_scores.numpy().tolist(), score_masks

    def _get_features(self, dataloader: DataLoader) -> torch.Tensor:
        self.feature_embedder.eval()
        features = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                features.append(self.feature_embedder(images))

        features = torch.cat(features, dim=0)
        return features

    def _get_coreset(self, trainloader: DataLoader, base_coreset=None) -> torch.Tensor:
        features = self._get_features(trainloader)
        coreset_features, _ = self.sampler.run(features, base_coreset=base_coreset)

        return coreset_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks