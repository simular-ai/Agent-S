from PIL import Image

# Open an image file
with Image.open('/data4/saaket/hallusion_bench/VS/map/0_1.png') as img:
    # Get the size of the image
    width, height = img.size

print(f'The image size is {width} x {height} pixels.')