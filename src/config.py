class Config:
    # Training parameters
    batch_size = 16
    lr = 0.0002
    beta1 = 0.5  # Adam optimizer beta1
    beta2 = 0.999
    num_epochs = 100
    device = "cuda"  

    # Model parameters
    img_channels = 3
    img_size = 128  
    latent_dim = 512
    feature_channels = [64, 128, 256, 512]  # Encoder / Decoder feature maps

    # Loss weights 
    lambda_adv = 1.0
    lambda_con = 50.0
    lambda_lat = 1.0

    # Misc
    save_interval = 10  # epochs
    log_interval = 100  # batches
