# imports.py
import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import asyncio
import aiohttp
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob
import numpy as np
import cv2
import json
import torch
import yaml
from dotenv import load_dotenv
from multiprocessing import Process
# External libraries for UI
import torchvision.transforms as transforms
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
from kivy.uix.popup import Popup
from transformers import AutoTokenizer, AutoModelForCausalLM
# Network and parsing libraries
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from requests_html import HTMLSession
from PIL import Image as PILImage
import torchvision.transforms as transforms
from PIL import Image
import exiftool
# Database operations
import sqlite3
import nltk
import spacy 
import torch
import psutil
import schedule
import time
# utils.py
import functools
import hashlib
import pandas as pd
from urllib.parse import urlparse
from io import BytesIO
from typing import Any, Dict, List, Tuple
from datetime import datetime
from functools import wraps
import threading
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import redis
import backoff
import concurrent.futures

# Environment-specific configurations
DATABASE_PATH = os.getenv('DATABASE_PATH', 'default.db')

