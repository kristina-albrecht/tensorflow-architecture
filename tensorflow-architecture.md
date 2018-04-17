---
title: "TensorFlow Architektur"
author:	[Jan Hofmeier, Kristina	Albrecht]
date: 2018-04-20
subject: "TensorFlow"
tags: [Hana, SSBM]
subtitle: "Software	Architektur"
titlepage: true
titlepage-color: 06386e
titlepage-text-color: FFFFFF
titlepage-rule-color: FFFFFF
titlepage-rule-height: 1
...

# TensorFlow-Architektur

## Was ist TensorFlow
--TensorFlow™ is an	open source	library	for	developing machine learning	applications. These	applications are implemented using graphs to organize the flow of operations and tensors for representing the data.	It offers an application programming interface (API) in	Python,	as well	as a lower level set of	functions implemented using	C++. It	provides a set of features to enable faster	prototyping	and	implementation of machine learning models and applications for highly heterogeneous	computing platforms.

## Stakeholders

- Forscher,	Studenten, Wissenschaftler
- Architekten und Software Engineure
- Entwickler
- Hardware Hersteller

## Anforderungen (Funktionale / Nicht-Funktionale)

- ML und DL	Funktionalitäten: Schnelle Rechenoperationen, Matrizen,	Lineare	Algebra	und	Statistik
- Flexibilität:	Forschung, Prototypen und Produktion
	--TensorFlow™ allows industrial researchers a faster product prototyping. It also provides	academic researchers with a	development	framework and a	community to discuss and support novel applications.
	--Provides	tools to assemble graphs for expressing	diverse	machine	learning models. New operations	can	be written in Python and low-level data	operators are implemented using	in C++.
- Performance: maximale	Effizienz und schnelle Berechnungen. 

- Portabilität
	--Runs	on CPUs, GPUs, desktop,	server,	or mobile computing	platforms. That	make it	very suitable in several fields	of application,	for	instance medical, finance, consumer	electronic,	etc.

## Anforderungsanalyse

-------------------------------------------------------------------------------
Faktor      Beschreibung        Flexibilität        Einfluss
Index                           Erweiterbarkeit
----------- ------------------- ------------------- ---------------------------
O1          Firmen              Flexibel/Variabel   Einfluss auf Architektur
-------------------------------------------------------------------------------


+---------------+---------------+---------------|-------------------------------+
| Faktor-Index  | Beschreibung  | Flexibilität  | Einfluss                      |
+---------------+---------------+---------------|-------------------------------+
| O1            | ...           |   Flexibel    | Einfluss auf die Architektur  |
+---------------+---------------+---------------|-------------------------------+
| T1            | ...           |   Fest        | ...                           |
+---------------+---------------+---------------|-------------------------------+
| P1            | ...           |   Variabel    | ...                           |
+---------------+---------------+---------------|-------------------------------+

## Architekturentwurf

### Kontext-Sicht

![Dependencies](img/Contextview.png){height=400px}

## Source-Code-Hierarchie
--TensorFlow™ 's root directory	at GitHub is organized in five main	subdirectories:	google,	tensorflow,	third-party, tools and util/python.	Additionally, the root directory provides information on how to	contribute to the project, and other relevant documents. In	figure 3, the source code hierarchy	is illustrated.

![Source-Code-Hierarchie](img/TensorFlowTree.png){height=400px}

###	Development-Sicht

###	Deployment-Sicht

