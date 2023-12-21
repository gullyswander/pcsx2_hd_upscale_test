"""pip install torch torchvision watchdog Pillow"""

import os
import torch
from PIL import Image
from model import RRDBNet  # This may vary based on your ESRGAN implementation
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class TextureHandler(FileSystemEventHandler):
    def __init__(self, model, device, output_folder):
        self.model = model
        self.device = device
        self.output_folder = output_folder

    def on_created(self, event):
        if not event.is_directory:
            input_path = event.src_path
            filename = os.path.basename(input_path)
            output_path = os.path.join(self.output_folder, filename)
            self.process_texture_file(input_path, output_path)

    def process_texture_file(self, input_path, output_path):
        if not os.path.exists(output_path):
            self.enhance_texture(input_path, output_path)

    def enhance_texture(self, input_path, output_path):
        img = Image.open(input_path).convert('RGB')
        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img).squeeze(0)

        output = transforms.ToPILImage()(output.cpu())
        output.save(output_path)

def load_model(model_path, device):
    model = RRDBNet(3, 3, 64, 23, gc=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    return model

def start_monitoring(input_folder, output_folder, model, device):
    event_handler = TextureHandler(model, device, output_folder)
    observer = Observer()
    observer.schedule(event_handler, input_folder, recursive=False)
    observer.start()

    print(f"Monitoring {input_folder} for new textures...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    import time
    import torchvision.transforms as transforms

    model_path = 'path_to_your_pretrained_esrgan_model.pth'
    texture_dump_folder = 'path_to_your_pcsx2_texture_dump_folder'
    output_folder = 'path_to_your_new_hd_textures_folder'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    start_monitoring(texture_dump_folder, output_folder, model, device)
