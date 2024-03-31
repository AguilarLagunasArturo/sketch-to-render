import os
import PIL.ImageOps
from PIL import Image

def process_image(
    image_path: str, output_path: str = None,
    thumbnail_size: tuple = (512, 768), threshold: int = 240, verbose: bool = True):

    save = True if output_path else False

    file_name = image_path.split('/')[-1].split('.')[0]
    image_file = Image.open(image_path)

    if image_file.mode == 'RGBA':
        # Create a blank white image with the same size
        white_background = Image.new('RGBA', image_file.size, (250, 250, 250, 250))
        # Paste the PNG image onto the white background, using itself as the mask
        white_background.paste(image_file, (0, 0), image_file)
        # Convert to RGB to discard the alpha channel
        image_file = white_background.convert('RGB')

    # image_file.thumbnail((768, 768))
    image_file.thumbnail(thumbnail_size)
    image_file_rs = image_file.convert('L')

    # Apply the threshold to each pixel
    image_file_bw = image_file_rs.point(lambda x: 0 if x > threshold else 255)

    # Save processed image
    if save:
        output_rs_file_path = os.path.join( output_path, f'rs_{file_name}.png' )
        output_bw_file_path = os.path.join( output_path, f'bw_{file_name}.png' )
        if verbose:
            print(f"[+] Saving: {image_path} -> {output_rs_file_path}")
            print(f"[+] Saving: {image_path} -> {output_bw_file_path}")
        image_file_rs.save(output_rs_file_path)
        image_file_bw.save(output_bw_file_path)

    return image_file_bw

def file_chooser(search_path):
    candidate_paths = [os.path.join(search_path, filename) for filename in os.listdir(search_path)]
    if len(candidate_paths) == 1:
        return candidate_paths[0]
    for i, candidate_path in enumerate(candidate_paths):
        print(f"[{i}] {candidate_path}")
    ix = int(input("[!] Choose a file: "))
    return candidate_paths[ix]
