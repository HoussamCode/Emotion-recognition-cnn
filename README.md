![Accuracy](screenshot/CNNss.PNG)

# Rapport du Projet : Classification des Émotions avec un Réseau de Neurones Convolutif (CNN)
## 1. Introduction

L’objectif de ce projet est de construire un modèle capable de reconnaître différentes émotions humaines à partir d’images.
Pour cela, nous utilisons un réseau de neurones convolutif (CNN), spécialement adapté à l'analyse d'images.

Le dataset contient des visages classés selon 7 émotions.
Le code permet de :

Préparer les données

Visualiser des images

Construire un CNN

Entraîner et valider

Évaluer

Tester les prédictions

## 2. Chargement du Dataset

Le dataset se trouve dans :

C:\Users\chhou\PycharmProjects\PythonProject3\emotions


Les images sont organisées par émotion (un dossier par classe).

Chargement avec image_dataset_from_directory() :

train_dataset = tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset="training")
validation_dataset = tf.keras.utils.image_dataset_from_directory(..., validation_split=0.2, subset="validation")
full_dataset = tf.keras.utils.image_dataset_from_directory(...)


80% → entraînement

20% → validation

15% → test (généré à partir du dataset complet reshufflé)

Les images sont redimensionnées à 48×48 pixels, batch size = 32.

## 3. Visualisation des Données

Avant l’entraînement, 9 images sont affichées :

Chaque image = un visage

Titre = émotion réelle

plt.imshow(images[i].numpy().astype("uint8"))
plt.title(class_names[labels[i]])


Ce contrôle visuel permet de confirmer la bonne importation du dataset.

## 4. Conception du Modèle CNN

Le modèle suit une architecture classique :

### 4.1- Normalisation
tf.keras.layers.Rescaling(1./255)

### 4.2- Convolutions + MaxPooling

Conv2D(32) → caractéristiques simples

Conv2D(64) → caractéristiques moyennes

Conv2D(128) → caractéristiques complexes

MaxPooling2D() entre chaque convolution

### 4.3- Couches Finales

Flatten()

Dense(128, activation="relu")

Dense(7, activation="softmax")

Cette architecture équilibre simplicité et performance.

## 5. Compilation et Entraînement

Le modèle est compilé avec :

Optimiseur : Adam

Loss : sparse_categorical_crossentropy

Métrique : accuracy

Entraînement sur 30 époques :

modelCNN.fit(train_dataset, epochs=30, validation_data=validation_dataset)


Suivi :

Précision d’entraînement

Précision en validation

## 6. Évaluation du Modèle

L’évaluation se fait sur le dataset de test :

modelCNN.evaluate(test_dataset, verbose=2)


Cette mesure reflète la performance sur des images jamais vues.

## 7. Prédictions Finales

Les prédictions sont réalisées avec :

prediction = modelCNN.predict(images)
predicted_classes = np.argmax(prediction, axis=1)


Puis affichage :

Vrai: <classe réelle> — Prédit: <classe prédite>


Cela permet de visualiser la qualité du modèle et identifier les confusions.
