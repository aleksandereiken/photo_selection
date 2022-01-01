from PIL import Image
from PIL.ExifTags import TAGS

def return_timestamp(imgs):
    '''
    :param imgs: a list of full paths to images
    :return: two dictonaries: first a dictonary of image name: timestamp of photo taken.
    Secondly a dictonary of images which Failed
    '''
    dict_timestamps = dict()
    errors_images = dict()
    for index, img in enumerate(imgs):
        try:
            image = Image.open(img)

            # extract EXIF data
            exifdata = image.getexif()

            # iterating over all EXIF data fields
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                if tag == "DateTime":
                    # print(tag)
                    tstamp = exifdata.get(tag_id)
                    # print(tstamp)
                    to_add = {img : tstamp}
                    dict_timestamps.update(to_add)
        except:
            to_add = {img: False}
            errors_images.update(to_add)
    return(dict_timestamps, errors_images)
