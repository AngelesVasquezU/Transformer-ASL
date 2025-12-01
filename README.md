# Vision Transformer (ViT)

## Reconocimiento de Vocales del Alfabeto en Lenguaje de Señas

Proyecto enfocado en la clasificación de **vocales del alfabeto en lenguaje de señas** utilizando el modelo **Vision Transformer (ViT)**.  

---

## Dataset

Para entrenar el modelo se utilizó el dataset **ASL Alphabet** de lenguaje de señas americano, disponible públicamente en Kaggle:

- Kaggle – ASL Alphabet: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

El dataset completo contiene 29 clases (todas las letras del alfabeto y algunas señas especiales), pero en este proyecto se utilizaron únicamente las **5 vocales**:

- `A`, `E`, `I`, `O`, `U`

Para reentrenar el modelo es necesario descargar el dataset desde Kaggle, descomprimirlo y colocar la carpeta `asl_alphabet_train/` en la raíz del proyecto (`Transformer-ASL/asl_alphabet_train/...`).

---

## Integrantes

- **Huicho Perez, Anthony**
- **Perales Estrada, Daniela**
- **Vásquez Pineda, María de los Angeles**
