import os
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
from datetime import datetime
import piexif
import re
register_heif_opener()

def convert_heic_to_jpeg(dir_of_interest):
        filenames = os.listdir(dir_of_interest)
        filenames_matched = [re.search("\.HEIC$|\.heic$", filename) for filename in filenames]

        # Extract files of interest
        HEIC_files = []
        for index, filename in enumerate(filenames_matched):
                if filename:
                        HEIC_files.append(filenames[index])

        # Convert files to jpg while keeping the timestamp
        for filename in HEIC_files:
                image = Image.open(dir_of_interest + "/" + filename)
                image_exif = image.getexif()
                if image_exif:
                        # Make a map with tag names
                        exif = { ExifTags.TAGS[k]: v for k, v in image_exif.items() if k in ExifTags.TAGS and type(v) is not bytes }
                        # Grab the date
                        date_obj = datetime.strptime(exif['DateTime'], '%Y:%m:%d %H:%M:%S')
                        # print(date_obj)
                        exif_dict = piexif.load(image.info["exif"])
                        exif_dict["0th"][piexif.ImageIFD.DateTime] = date_obj.strftime("%Y:%m:%d %H:%M:%S")
                        exif_dict["0th"][piexif.ImageIFD.Orientation] = 1
                        exif_bytes = piexif.dump(exif_dict)
                        image.save(dir_of_interest + "/" + os.path.splitext(filename)[0] + ".jpg", "jpeg", exif=exif_bytes)
                else:
                        print(f"Unable to get date from exif for {filename}")

