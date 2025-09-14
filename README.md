🗺️ OverlayGPX
🎬 Transformez vos fichiers GPX en vidéos animées

OverlayGPX est une application Python qui transforme un fichier GPX en une vidéo MP4 avec carte animée, tracé GPS, et données sportives synchronisées (altitude, vitesse, allure, fréquence cardiaque…).

  ✨ Fonctionnalités

🗺️ Carte de fond avec trace GPS animée (OpenStreetMap, CyclOSM, IGN Satellite…)
⛰️ Profil d’altitude

🚴 Profil de vitesse

⏱️ Profil d’allure (min/km)

❤️ Profil de fréquence cardiaque

⚡ Jauge de vitesse en temps réel

📝 Bloc d’infos en direct : vitesse, altitude, heure, pente, allure, FC

🧭 Flèche nord et rotation dynamique de la carte

🎞️ Export en vidéo MP4 (codec H.264)

🖥️ Interface graphique Tkinter intuitive :

Choix de la résolution et des FPS

Choix du style de carte et niveau de zoom (1 à 12)

Personnalisation complète des couleurs

Disposition libre de chaque élément

📸 Aperçu instantané de la première image avant rendu complet

## Fonds de carte

Certains fournisseurs limitent le niveau de zoom ou imposent des règles d'accès.
Le programme ajuste automatiquement le niveau de zoom pour éviter les tuiles manquantes
et envoie un `User-Agent` explicite.

- OpenSnowMap (relief et pistes) : zoom ≤ 18
- OpenStreetMap, CyclOSM, IGN : zoom ≤ 19

La couche "OpenSnowMap" correspond à un overlay de pistes transparent superposé au
fond "OpenSnowMap Relief".
