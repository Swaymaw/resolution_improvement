import os
from PIL import Image 

if not os.path.exists('data/lr_images'):
    os.makedirs('data/lr_images')
if not os.path.exists('data/hr_images'):
    os.makedirs('data/hr_images')

for img in os.listdir('original_data/'):
    hr_image = Image.open(os.path.join('original_data', img)).resize((256, 256)).convert('RGB')

    lr_image = Image.open(os.path.join('original_data/', img)).resize((64, 64)).convert('RGB')
    hr_image.save(f'data/hr_images/{img}')
    lr_image.save(f'data/lr_images/{img}')