# I. Clustering k-means
## 1) Implémentation de l'algo 
- Le cours propose un algorithme idéal dans lequel la condition de fin est l'égalité entre les 
  centroïdes entre deux itérations. Cette approche peut poser problème si l'on a un grand nombre 
  de points, car l'algorithme peut effectuer un grand nombre d'itérations pour un changement 
  ayant peu d'impact sur le résultat final.
- C'est pourquoi l'on propose une fonction remplaçant cette condition par deux paramètres _
  (nombre max d'itérations, tolérance de convergence)_ pour arrêter l'algorithme avec des résultats
  satisfaisants au niveau des résultats comme des performances.
