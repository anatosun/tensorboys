from svgpathtools import svg2paths, paths2svg, Path
import numpy as np
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu as otsu
import cv2
import os
from sklearn.preprocessing import minmax_scale
from scipy.spatial.distance import cdist
from pyts.metrics import dtw


# getting the sliced images with the corresponding label from the initial image and the svg

class Slicer:
    def __init__(self, images: tuple, frames: tuple) -> None:
        self.images = self.__fileList(images[0], images[1])
        self.frames = self.__fileList(frames[0], frames[1])
        assert(len(self.images) == len(self.frames)
               ), f"Number of images {len(self.images)} and frames {len(self.frames)} must be equal"
        assert(len(self.frames) != 0), f"No images nor frames found"
        self.documents = [cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                          for image in self.images]

    def __fileList(self, path: str, ext: str) -> list:
        matches = []
        for root, _, filenames in os.walk(os.path.join(path)):
            for filename in filenames:
                if filename.endswith((ext)):
                    matches.append(os.path.join(root, filename))
        return matches

    def __get_frame(self, image, image_number: str, path: Path, id: int, width=600, height=120, save=False, savepath='./output') -> np.array:
        frame = paths2svg.big_bounding_box(path)
        frame = tuple(map(int, frame))
        document_crop = image[frame[2]:frame[3], frame[0]:frame[1]]
        tresh = otsu(document_crop)
        scan_crop_logic = document_crop < tresh
        original_width = int(frame[1]-frame[0])
        original_height = int(frame[3]-frame[2])
        box = Image.new('1', (original_width, original_height), "black")
        box_image = ImageDraw.Draw(box)
        box_image.polygon([(int(edge.point(0).real)-frame[0],
                            int(edge.point(0).imag)-frame[2]) for edge in path], fill="white", outline=None)
        img = np.logical_and(scan_crop_logic, box)
        img = np.invert(img)
        img = img.astype(float)
        img = cv2.resize(src=img, dsize=(width, height),
                         interpolation=cv2.INTER_NEAREST)
        if save:
            path = os.path.join(savepath, image_number)
            if not os.path.exists(path):
                os.makedirs(path)
            Image.fromarray(img > 0).save(
                os.path.join(path, f"{id}.png"))

        return np.asarray(img)

    def get_frames(self, save=False, savepath='./output') -> tuple:
        ids = np.array([attribute['id'] for _, frame in zip(
            self.images, self.frames) for _, attribute in zip(*svg2paths(frame))])
        frames = np.array([self.__get_frame(image=document, image_number=os.path.splitext(os.path.basename(image))[
            0], path=path, id=attribute['id'], save=save, savepath=savepath) for image, document, frame in zip(self.images, self.documents, self.frames) for path, attribute in zip(*svg2paths(frame))])
        return ids, frames


def main():
    ids, frames = Slicer(images=("./data/images", ".jpg"),
                         frames=("./data/ground-truth/locations", ".svg")).get_frames(save=True, savepath="./data/output/")


def lower_contour(slice):

    index = np.where(slice <= 0.5)[0]

    if len(index) == 0:
        return 0
    else:
        return np.max(index)


def upper_contour(slice):

    index = np.where(slice <= 0.5)[0]

    if len(index) == 0:
        return len(slice)
    else:
        return np.min(index)


def fraction_black(slice):

    number = len(np.where(slice <= 0.5)[0])

    if number == 0:
        return 0.0
    else:
        return number/len(slice)


def fraction_black_between(slice):

    number = len(np.where(slice <= 0.5)[0])

    if number == 0:
        return 0.0
    else:

        index_min = np.min(np.where(slice <= 0.5))
        index_max = np.max(np.where(slice <= 0.5))

        return fraction_black(slice[index_min:index_max+1])


def transition_black_white(slice):

    counter = 0
    was_on_black = False

    for i in range(len(slice)):

        if slice[i] <= 0.5 and was_on_black == False:
            was_on_black = True

        elif slice[i] > 0.5 and was_on_black == True:
            was_on_black = False
            counter += 1

    return counter


# Compute the "time serie", the features for each slice of the image by sliding one pixel after the other
def compute_features_vector(image, normalize=True):

    rep = []
    for i in range(image.shape[1]):
        feature_vector = [lower_contour(image[:, i]), upper_contour(image[:, i]), fraction_black(
            image[:, i]), fraction_black_between(image[:, i]), transition_black_white(image[:, i])]
        rep.append(feature_vector)

    array = np.asarray(rep)

    if normalize:
        array = minmax_scale(array, feature_range=(0, 1), axis=0)

    return array


def compute_dtw(image1, image2, windows_size=0.5, normalize=True):

    feature_vector_1 = compute_features_vector(image1, normalize)
    feature_vector_2 = compute_features_vector(image2, normalize)

    dist_matrix = cdist(feature_vector_1, feature_vector_2)

    dtw_cost = dtw(precomputed_cost=dist_matrix, dist="precomputed",
                   method="sakoechiba", options={"window_size": windows_size})

    return dtw_cost


if __name__ == "__main__":
    main()
