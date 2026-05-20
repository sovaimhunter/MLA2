import kagglehub

# Download latest version
path = kagglehub.dataset_download("christianlillelund/csgo-round-winner-classification")

print("Path to dataset files:", path)
