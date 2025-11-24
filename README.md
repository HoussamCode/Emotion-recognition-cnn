![Accuracy](screenshot/CNNss.PNG)

Rapport du Projet : Classification des Émotions avec un Réseau de Neurones Convolutif (CNN)
1. Introduction
L’objectif de ce projet est de construire un modèle capable de reconnaître différentes émotions humaines à partir d’images. Pour cela, nous utilisons un réseau de neurones convolutif (CNN), un type de modèle particulièrement adapté à l’analyse d’images.
Le dataset utilisé contient des images de visages classées selon 7 émotions différentes. Le code mis en place permet de préparer les données, visualiser des exemples, construire le modèle, l’entraîner, l’évaluer et tester les prédictions finales.
________________________________________
2. Chargement du Dataset
Le dataset se trouve dans le répertoire :
C:\Users\chhou\PycharmProjects\PythonProject3\emotions
Les images sont organisées en sous-dossiers, un dossier par émotion.
Le code utilise image_dataset_from_directory() pour créer automatiquement un dataset avec :
•	Un dataset d'entraînement (80%)
•	Un dataset de validation (20%)
•	Un dataset complet (reshufflé) permettant de créer un dataset de test (15%)
train_dataset = tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset="training")
validation_dataset = tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset="validation")
full_dataset = tf.keras.utils.image_dataset_from_directory(...)
Les images sont redimensionnées à 48×48 pixels avec un batch size de 32.
________________________________________
3. Visualisation des Données
Avant l'entraînement, un échantillon d’images est affiché :
•	9 images sont montrées
•	Chaque image montre un visage annoté avec son émotion réelle
Cela permet d’avoir un aperçu du dataset et de vérifier que l’importation est correcte.
plt.imshow(images[i].numpy().astype("uint8"))
plt.title(class_names[labels[i]])
________________________________________
4. Conception du Modèle CNN
Le modèle construit suit une architecture classique composée de :
🔹 1. Normalisation :
tf.keras.layers.Rescaling(1./255)
Pour mettre les valeurs de pixels entre 0 et 1.
🔹 2. Trois blocs Convolution + MaxPooling :
•	Conv2D(32 filtres) → extraction de caractéristiques simples (bords, textures)
•	Conv2D(64 filtres) → extraction intermédiaire
•	Conv2D(128 filtres) → extraction de caractéristiques plus complexes
•	Après chaque convolution, une couche MaxPooling2D réduit la dimension.
🔹 3. Couches Fully Connected :
•	Flatten() aplati les cartes de features
•	Dense(128, relu) → réseau dense intermédiaire
•	Dense(7, softmax) → sortie à 7 classes (une par émotion)
Cette architecture balance bien simplicité et performance.
________________________________________
5. Compilation et Entraînement
Le modèle est compilé avec :
•	Optimiseur : Adam
•	Loss : sparse_categorical_crossentropy
•	Métrique : accuracy
Puis entraîné pendant 30 époques :
modelCNN.fit(train_dataset, epochs=30, validation_data=validation_dataset)
Cela permet de :
•	Surveiller la précision d’entraînement
•	Vérifier la généralisation via la validation
________________________________________
6. Évaluation du Modèle
Le dataset de test extrait du dataset complet est utilisé :
modelCNN.evaluate(test_dataset, verbose=2)
Cette étape permet d’obtenir une mesure objective des performances du modèle sur des images jamais vues.
________________________________________
7. Prédictions Finales
Le modèle réalise ensuite des prédictions sur un batch du dataset de test :
•	On applique le modèle pour obtenir un vecteur de probabilités
•	On prend la classe avec argmax
•	On affiche pour chaque image :
Vrai: <classe réelle> — Prédit: <classe prédite>
Cela permet de vérifier la qualité des prédictions et d’identifier les éventuelles confusions.

