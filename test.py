from vectorizers.dino import Dino
from image_preprocessing.dino_preprocessing import DinoPreprocessor
from image_feature_extraction import ExtractFeatureMethod
from PIL import Image

if __name__ == "__main__":
    '''dino_base = Dino("small")
    img =  Image.open("data/test_image_headmind/image-20210928-102718-2474636a.jpg")
    print(dino_base(img).shape)'''

    preprocessor = DinoPreprocessor()
    vectorizer = Dino("small")
    embeddings_pipeline = ExtractFeatureMethod(preprocessing_function=preprocessor, vectorization_function=vectorizer)
    embeddings_pipeline.run_on_paths(["data/DAM"], save_folder_name= "dino-test", save_unique_df=True)