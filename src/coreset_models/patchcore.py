

class PatchCore(BaseCore):

    def __init__(
        self,
        feature_embedder: FeatureEmbedder,
        device,
        imagesize,
        coreset_ratio=0.01,
        greedy_proj_dim=128,
        faiss_on_gpu=True,
        faiss_num_workers=8,
        sampler_on_gpu=True,
        anomaly_score_num_nn=1,
        save_dir_path=None,
        brute=True,
    ):
        super(PatchCore, self).__init__()

        self.feature_embedder = feature_embedder

        self.greedy_coreset_sampler = GreedyCoresetSampler(
            percentage=coreset_ratio,
            dimension_to_project_features_to=greedy_proj_dim,
            device=device if sampler_on_gpu else torch.device("cpu"),
            brute=brute,
        )

        self.faiss_on_gpu = faiss_on_gpu
        self.faiss_num_workers = faiss_num_workers
        self.device = device
        self.anomaly_score_num_nn = anomaly_score_num_nn

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

        gc.collect()
        torch.cuda.empty_cache()

        if set_predictor:

            nn_method = FaissNN(
                self.faiss_on_gpu, self.faiss_num_workers, device=self.device.index
            )
            self.anomaly_scorer = NearestNeighbourScorer(
                n_nearest_neighbours=self.anomaly_score_num_nn, nn_method=nn_method
            )

            self.anomaly_scorer.fit([coreset_features])

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

    def _get_features(
        self, dataloader: DataLoader
    ) -> torch.Tensor:

        self.feature_embedder.eval()
        features = []
        with tqdm(
            dataloader, desc="Computing support features...", leave=False
        ) as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    images = data["image"]
                
                _features = self.feature_embedder(images)
                features.append(_features)

        features = torch.cat(features, dim=0)

        return features

    def _get_coreset(self, trainloader: DataLoader) -> torch.Tensor:
        
        features = self._get_features(trainloader)
        coreset_features, _ = self.greedy_coreset_sampler.run(features)

        return coreset_features

    def predict(self, images) -> np.ndarray:
        with torch.no_grad():
            features = self.feature_embedder(images)
        image_scores, score_masks = self._get_scores(np.array(features))

        return image_scores, score_masks