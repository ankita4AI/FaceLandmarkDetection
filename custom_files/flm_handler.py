import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler

class FLMHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms. CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])
        ])

    def preprocess_one_request(self, req):
        processed_images = []
        # get image from the request
         # create a stream from the encoded image
        images = list(req.values())
        for image in images:
            processed_images.append(self.preprocess_one_image(image))
        return torch.cat(processed_images)

    def preprocess_one_image(self, image):
        image = Image.open(io.BytesIO(image)).convert("L")
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_request(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, x):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        outs = self.model.forward(x)
        # probs = F.softmax(outs, dim=1) 
        # preds = torch.argmax(probs, dim=1)
        return outs

    def postprocess(self, preds):
        """
        Given the data from .inference, postprocess the output.
        In our case, we get the human readable label from the mapping 
        file and return a json. Keep in mind that the reply must always
        be an array since we are returning a batch of responses.
        """
        res = []
        # pres has size [BATCH_SIZE, 1]
        # convert it to list
        preds = preds.cpu().tolist()
        # for pred in preds:
        #     label = self.mapping[str(pred)][1]
        #     res.append({'label' : label, 'index': pred })
        # here length of preds would be number of images in one request, to make it consistent with a single request
        return [preds]