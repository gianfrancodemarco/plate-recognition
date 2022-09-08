def from_not_annotated_to_training_set(images):
    for image in images:
        annotations = pd.read_csv(annotations_path)
        next_index = annotations['name'].tolist()[-1] + 1

        print(image)
        img = cv2.imread(os.path.join(not_annotated_images_path, str(image)))
        img = current_function(img)
        img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

        print(os.path.join(not_annotated_images_path, str(image)))
        ny = model.predict(img)[0]
        print(ny)
        annotations.loc[len(annotations.index)] = [next_index, int(ny[0]), int(ny[1]), int(ny[2]), int(ny[3])]

        src = os.path.join(not_annotated_images_path, str(image))
        dst = os.path.join(images_path, str(next_index) + '.jpg')
        !cp $src $dst

        dst = os.path.join('/content/gdrive/MyDrive/final_dataset/images', str(next_index) + '.jpg')
        !cp $src $dst

        !rm $src
        src = os.path.join('/content/gdrive/MyDrive/final_dataset/not_annotated_images', str(image))
        !rm $src

        with open(annotations_path, 'w') as f:
            annotations.to_csv(f, index=False)

        with open(os.path.join(gdrive_path, 'annotations.csv'), 'w') as f:
            annotations.to_csv(f, index=False)


upgrade_images = [

    '366.jpg',
    '17.jpg',
    '306.jpg'

]

from_not_annotated_to_training_set(upgrade_images)
