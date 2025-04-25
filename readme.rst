.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/HaqTetsuya/ChatbotPerpusBipa/blob/main/IndobertPerpusChatbot.ipynb
   :alt: train the model


.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/HaqTetsuya/ChatbotPerpusBipa/blob/main/books.ipynb
   :alt: train the model


modelnya belum jadi



ChatbotPerpusBipa
=================

ChatbotPerpusBipa is a chatbot designed for the library system of STMIK BINA PATRIA. It is built using **CodeIgniter 3** and **IndoBERT** for natural language processing. The repository contains scripts and models to create and deploy the chatbot.

---

Features
--------

- Chatbot system for library for STMIK BINA PATRIA
- Built with CodeIgniter 3 for backend
- Utilizes IndoBERT for NLP tasks (Indonesian language support)

---

Setup Instructions
------------------

1. Clone the repository:

::

    git clone https://github.com/HaqTetsuya/ChatbotPerpusBipa.git
    cd ChatbotPerpusBipa

2. Install required Python libraries:

::

    pip install -r requirements.txt

3. Set up the CodeIgniter framework:

   - Place the PHP files in a web server directory (e.g., ``htdocs`` for XAMPP).
   - Configure the database in ``application/config/database.php``.

4. Ensure your environment is configured for PHP and Python integration.

---

Training the Model
------------------

Before running the chatbot, you need to train the IndoBERT model. Use the provided Colab notebook for training:

.. image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/HaqTetsuya/ChatbotPerpusBipa/blob/main/IndobertPerpusChatbot.ipynb
   :alt: Open in Colab

1. Open the Colab notebook by clicking the badge above.
2. Follow the instructions in the notebook to train the model:

   - Upload your dataset.
   - Preprocess the data.
   - Train and save the model.

3. Download the trained model and place it in the appropriate directory in the project: ``/py/model/``.

---

Usage
-----

1. Start the web server to launch the chatbot interface.
2. Load the trained model into the chatbot system.
3. Access the chatbot via your browser (e.g., ``http://localhost/ChatbotPerpusBipa/chat4``).

---

License
-------

This project is licensed under the MIT License. See the ``LICENSE`` file for more details.

---

SEMENTARA
-------
https://drive.google.com/drive/folders/1O5invCEiVLuQQHBDlQy1gzZpgg5ZcHrj?usp=sharing
