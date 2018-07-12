from .steppy.base import BaseTransformer


class GoogleAiLabelEncoder(BaseTransformer):
    def transform(self, annotations, annotations_human_labels):
        print(annotations.head())
        exit()
        images_with_scores = []
        for image, score in tqdm(zip(images, scores)):
            images_with_scores.append((image, score))
        return {'annotations': images_with_scores,
                'annotations_human_labels': images_with_scores}


class GoogleAiLabelDecoder(BaseTransformer):
    def transform(self, encoder):
        images_with_scores = []
        for image, score in tqdm(zip(images, scores)):
            images_with_scores.append((image, score))
        return {'images_with_scores': images_with_scores}
