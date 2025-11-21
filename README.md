# 🗺️ OverlayGPX (version web)

## 🎬 Présentation
Cette version de **OverlayGPX** s'exécute directement dans le navigateur : chargez un fichier GPX, visualisez le tracé sur une carte Leaflet, suivez les métriques synchronisées (vitesse, altitude, allure, fréquence cardiaque, pente) et exportez l'animation en vidéo WebM grâce à l'API `MediaRecorder`.

## ✨ Fonctionnalités principales
- **Carte animée** : prise en charge de plusieurs fournisseurs de tuiles (OpenStreetMap, ESRI Satellite, CyclOSM, etc.), affichage du tracé complet et d'un marqueur animé.
- **Graphiques synchronisés** : altitude, vitesse, allure (min/km) et fréquence cardiaque avec lissage configurable.
- **Indicateurs temps réel** : distance cumulée, heure locale, pente instantanée et jauge de vitesse.
- **Export vidéo** : capture de la zone de rendu (`captureStream`) pour produire un fichier WebM directement depuis l'interface.

## 🚀 Utilisation
1. Ouvrir `index.html` dans un navigateur moderne (Chrome/Edge/Firefox).
2. Importer un fichier `.gpx` via le bouton « Charger un fichier GPX ».
3. Ajuster la durée du clip, le lissage des graphes et le style de carte.
4. Cliquer sur **Démarrer** pour lancer l'animation, puis éventuellement sur **Exporter en WebM** pour sauvegarder la capture.

## 📁 Structure du projet
- `index.html` : layout principal, contrôles et inclusion des dépendances CDN (Leaflet, Chart.js).
- `styles.css` : thème sombre, grilles d'informations et jauge.
- `app.js` : parseur GPX, calculs de métriques, animation cartographique, graphes et export vidéo.
- `OverlayGPX_V1.py` : version Python historique conservée pour référence.

## 📝 Notes
- Les fournisseurs de tuiles en ligne nécessitent une connexion Internet lors de l'utilisation.
- L'export WebM repose sur `MediaRecorder`; certaines versions de Safari peuvent être limitées.
