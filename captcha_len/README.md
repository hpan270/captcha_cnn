This folder contains the code for a simple cnn model to predict the lenght of charactors in a captcha image
- genCaptchas.py: call this py first when there is no existing captcha images as training set
- captcha_len_identifier.py: main code for train captcha-length cnn and evaluate the accuracy of captcha length
- predict_CL.py: load the weight and json file to re-structure the cnn, and do prediction
- load_data.py: load image (e.g. standardize image vector)
- captcha_params.py: initial parameter setting
- model_cap_len_iden.json: the cnn structure file
- best.out: an log example when training the captcha-length cnn
