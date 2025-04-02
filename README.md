## Car object detection

Projecto sobre la deteccion de autos usando un modelo de Inteligencia Artificial llamado YOLO (You Only Look Once) desarrollado por Ultralytics, e integrando el proyecto con librerias como:


- Numpy
- Opencv
- sort
- cv2
- cvzone


Se implementan una mascara para detectar solo unicamente los autos en la carretera, posterior se realiza un trakeo de cada uno de estos autos con **sort**, frame por frame, a cada auto detectado y trakeado se le asigna un ID unico para poder realizar el conteo de cada auto que cruza y poder imprimir el conteo en pantalla. 


En la consola se imprime la clase del objeto, el nivel de confidencia, la posicion y el ID de cada auto detectado.


https://github.com/user-attachments/assets/4d31fb3f-2c28-4533-af5c-18976f9784c7
