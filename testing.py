def predict(path):
    '''
    The model variable is named cnn_model
    '''
    mean = 103.303632997174
    std = 95.32723806245423
    image = cv2.imread(path)
    image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)[None]
    return (cnn_model(image.cuda()).item() * std) + mean