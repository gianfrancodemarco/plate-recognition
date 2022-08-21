def from_not_annotated_to_training_set(images):
    for image in images:
        annotations = pd.read_csv(annotations_path)
        next_index = annotations['name'].tolist()[-1] + 1

        print(image)
        img = cv2.imread(os.path.join(not_annotated_images_path, str(image)))
        # img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = get_laplacian(img)
        img = img.reshape(1, img.shape[0], img.shape[1])

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

    '227.jpg',
    '404.jpg',
    '247.jpg',
    '270.jpg',
    '146.jpg',
    '160.jpg',
    '119.jpg',
    '27.jpg',
    '126.jpg',
    '446.jpg',
    '91.jpg',
    '285.jpg',
    '151.jpg',
    '314.jpg',
    '463.jpg',
    '26.jpg',
    '406.jpg',
    '317.jpg',
    '45.jpg',
    '279.jpg',
    '264.jpg',

]

from_not_annotated_to_training_set(upgrade_images)
