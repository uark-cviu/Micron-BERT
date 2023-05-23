import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torchvision import transforms


img_size = 224
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def image_to_tensor(image):
    x = torch.tensor(image)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum("nhwc->nchw", x).float()
    return x.cuda()


def preprocess(image, img_size):
    image = cv2.resize(image, (img_size, img_size)) / 255
    image = image - imagenet_mean
    image = image / imagenet_std
    return image


def load_image(frame_path, img_size):
    # print(frame_path)
    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return preprocess(image, img_size)


def get_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    args = checkpoint["args"]
    print(args)
    print(checkpoint["epoch"])

    import models.mae as mae_dict

    if "mae" in args.model_name:
        import models.mae as mae_dict

        model = mae_dict.__dict__[args.model_name](
            has_decoder=args.has_decoder,
            aux_cls=args.aux_cls,
            img_size=args.img_size,
            att_loss=args.att_loss,
            diag_att=args.diag_att,
            # DINO params,
            enable_dino=args.enable_dino,
            out_dim=args.out_dim,
            local_crops_number=args.local_crops_number,
            warmup_teacher_temp=args.warmup_teacher_temp,
            teacher_temp=args.teacher_temp,
            warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
            epochs=args.epochs,
        )

    model_state_dict = {}
    for k, v in checkpoint["model"].items():
        model_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(model_state_dict)
    return model, args


checkpoint = 'checkpoints/CASME2-is224-p8-b16-ep200.pth'
your_image_path = 'image.jpg'
model = get_model(checkpoint)[0].cuda()
model.eval()

# To extract features
image = load_image(your_image_path)
image_tensor = image_to_tensor(image)

with torch.no_grad():
    features = model.extract_features(image_tensor)
    # Use this features for finetunning.
