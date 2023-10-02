# **Entregable_productivizacion**

Este trabajo consiste en la subida a producción de un modelo predictivo, concretamente para clasificar noticias reales de otras "fake".

Las fases del proyectos son las siguientes:
- **Creación de la web (API)** para realizar peticiones al modelo, online.
- **Git** para organizar el código simulando un proyecto real.
- **Subida a Cloud** para poder acceder al modelo desde cualquier parte del mundo.
- **Spark** para simular peticiones a una base de datos real.

Este repositorio contendrá tres ramas principales:
- **main**: Rama de producción
- **HotFix**: Rama para pequeños cambios en producción
- **Develop**: Rama sobre la que se realizarán los merge de las distintas funcionalidades (flask, AWS, APIs, etc.)

En primer lugar, se deberá realizar un forkeo y clonado del repositorio. 

Cada desarrollador creará una rama adicional que se nombrará como Feature_nombre (ie: Feature_Flask, feature_API) y sobre la cual trabajará y realizará los distintos commits.

Una vez el commit tenga la funcionalidad deseada, se realizará, previa autorización, el merge con la rama develop. 

Una vez todas las funcionalidades hayan sido añadidas a la rama Develop, se realizará un merge con la rama HotFix, sobre la cual se realizarán pequeños cambios para optimizar o mejorar aspectos mínimos. 

Cuando el funcionamiento sea el correcto, se realizará un último merge con main, para tener el código final que se subirá a producción. 
