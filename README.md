# 🗺️ OverlayGPX

## 🎬 Présentation
OverlayGPX est une application Python qui transforme un fichier GPX en une vidéo MP4 avec carte animée, tracé GPS et données sportives synchronisées (altitude, vitesse, allure, fréquence cardiaque, etc.). L'objectif est d'offrir un rendu prêt à partager, combinant des informations de navigation et des métriques d'entraînement dans une interface entièrement personnalisable.

## ✨ Fonctionnalités détaillées
- **Carte animée** : téléchargement automatique de tuiles (OpenStreetMap, CyclOSM, Satellite ESRI, etc.), centrage dynamique sur le parcours, affichage de la trace complète et progression du point courant.
- **Graphiques synchronisés** : profils d'altitude, de vitesse, d'allure (min/km) et de fréquence cardiaque, avec lissage configurable pour un rendu fluide.
- **Indicateurs temps réel** : bloc d'informations regroupant vitesse instantanée, altitude, heure, pente, allure et fréquence cardiaque, ainsi qu'une jauge de vitesse linéaire ou circulaire.
- **Orientation et contexte** : boussole animée indiquant le nord, calcul de la rotation de la carte et affichage optionnel d'un ruban directionnel.
- **Export vidéo** : rendu en MP4 (codec H.264) avec choix de la résolution, des FPS, de la durée du clip et du style visuel pour chaque élément affiché.
- **Interface Tkinter** : prévisualisation immédiate de la première image, positionnement libre des éléments par glisser-déposer et sauvegarde des configurations d'affichage.

## 🛠️ Dépendances Python
OverlayGPX repose sur les bibliothèques tierces suivantes :

| Fonction | Bibliothèque | Installation |
|----------|--------------|--------------|
| Lecture GPX | `gpxpy` | `pip install gpxpy` |
| Calculs scientifiques | `numpy`, `scipy` | `pip install numpy scipy` |
| Fuseaux horaires | `pytz` | `pip install pytz` |
| Manipulation d'images | `Pillow`, `imageio` | `pip install pillow imageio` |
| Tuiles cartographiques *(optionnel)* | `staticmap` | `pip install staticmap` |

> ℹ️ `staticmap` est uniquement requis si vous souhaitez générer les cartes de fond. Sans cette dépendance, la vidéo peut être produite mais sans couche cartographique.

Les autres modules utilisés (`tkinter`, `json`, `math`, `threading`, etc.) font partie de la bibliothèque standard Python.

## 🚀 Utilisation rapide
1. Installer les dépendances listées ci-dessus.
2. Lancer le script principal :
   ```bash
   python OverlayGPX_V1.py
   ```
3. Charger un fichier GPX via l'interface, ajuster les options (style de carte, couleurs, disposition, FPS) et lancer le rendu vidéo.

## 📄 Licence
Ce projet est distribué sous licence MIT. Consultez le fichier `LICENSE` si disponible pour plus de détails.
