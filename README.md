# LesAutruches
On dispose d’un dataset de produit Dior sur fond neutre, qu’on désignera par dataset de référence, ainsi que d’images test.
L’idée générale est d’utiliser un modèle pré-entrainé tel que Mask R-CNN ou autre, afin de segmenter l’image, et identifier les pixels correspondant au produit (sur l’image test et le dataset de référence)
Un réseau traduit ensuite ces pixels par un vecteur de contexte. Ce vecteur est préenregistré pour les images du dataset de référence.
Lorsqu’on reçoit un image test, son vecteur de contexte est calculé et on retourne les vecteurs les plus similaires.