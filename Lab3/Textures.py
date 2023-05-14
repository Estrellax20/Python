import os
from PIL import Image

initial_photos = 'Images'  # name of the directory containing the input photos
texture_samples = 'T_images'  # the name of the directory where the iter_texture samples will be saved
new_size_tSample = 128  # the size of each iter_texture S (128x128 pixels)

for name_of_image in os.listdir(initial_photos):
    # extracting the filename without the file extension from the name_of_image variable
    image_without_extension = name_of_image.split(".")[0]
    new_image_path = os.path.join(initial_photos, name_of_image)  # create new_path to image
    new_text_path = os.path.join(texture_samples, image_without_extension)  # create new_path to iter_texture directory

    try:  # try to create directory for Text_samples
        os.mkdir(new_text_path)
    except FileExistsError:
        print('Error: Directory for T_images already exists')

    open_image = Image.open(new_image_path)  # open image; assigns the opened image object to the open_image
    # line calculates the number of iter_texture cuts in the x-direction and y-direction; divides the width of the open_image by the
    # new_size_tSample, which represents the desired size of the iter_texture samples
    number_of_xdirection_cuts = round(open_image.size[0] / new_size_tSample) - 1  # result is rounded to the nearest integer and assigned to the variable
    number_of_ydirection_cuts = round(open_image.size[1] / new_size_tSample) - 1

    for x in range(
            number_of_xdirection_cuts):  # set up nested loops to iterate over the x and y coordinates of the iter_texture samples
        for y in range(number_of_ydirection_cuts):
            # creates a unique name for each iter_texture S by combining image_without_extension, the current
            # x index, the underscore character (_), the current y index, and the ".jpg" file extension
            name_for_tSample = image_without_extension + str(x) + '_' + str(y) + '.jpg'
            # crops a section of the open_image using the coordinates calculated based on the current x and y indices
            # crop() - specifying the coordinates of the top-left and bottom-right corners of the desired section
            cropped_image_by_using_coordinates = open_image.crop((x * new_size_tSample, y * new_size_tSample, (x + 1) * new_size_tSample, (y + 1) * new_size_tSample))
            # creates the new_path where the iter_texture S will be saved by combining the texture_samples directory new_path, image_without_extension
            # and the name_for_tSample
            new_path = os.path.join(texture_samples, image_without_extension, name_for_tSample)
            # saves the cropped iter_texture S as a new JPEG file at the specified new_path
            cropped_image_by_using_coordinates.save(new_path)

            # (x * new_size_tSample, y * new_size_tSample) -> top-left corner of the cropping region
            # (x + 1) and (y + 1) are used to calculate the bottom-right corner of the cropping region by obtaining the next x and y coordinates
