import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections import Counter
from pylab import savefig
import cv2 as cv
from flask import send_file
import os
import random
import pymongo
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
from flask import Flask, render_template, Response
import csv


def calculate_totals(target_year, target_month):
    # Koneksi ke MongoDB
    client = pymongo.MongoClient("mongodb+srv://aininurulazizah:aininurulazizah@cluster0.cmotjqx.mongodb.net/?retryWrites=true&w=majority")

    # Memilih database
    db = client.get_database('tweets_db')

    # Mengambil koleksi
    records = db.data_tweets

    # Query untuk mencari data dengan tahun dan bulan tertentu
    date_query = {
        "$expr": {
            "$and": [
                {"$eq": [{"$year": {"$toDate": "$post_date"}}, target_year]},
                {"$eq": [{"$month": {"$toDate": "$post_date"}}, target_month]}
            ]
        }
    }

    # Proyeksi untuk hanya mengambil kolom yang diperlukan
    projection = {"category": 1, "quote_count": 1, "reply_count": 1, "retweet_count": 1, "like_count": 1, "_id": 0}

    # Membuat pipeline agregasi
    pipeline = [
        {"$match": date_query},
        {"$project": projection}
    ]

    # Melakukan agregasi
    result = list(records.aggregate(pipeline))

    # Membuat dictionary untuk menyimpan total untuk setiap kategori
    category_totals = {}

    for doc in result:
        category = doc.get("category")
        if category not in category_totals:
            category_totals[category] = {
                "total_quote_count": 0,
                "total_reply_count": 0,
                "total_retweet_count": 0,
                "total_like_count": 0,
                "total_engagement": 0,  # Add the total_engagement field
            }

        category_totals[category]["total_quote_count"] += doc.get("quote_count", 0)
        category_totals[category]["total_reply_count"] += doc.get("reply_count", 0)
        category_totals[category]["total_retweet_count"] += doc.get("retweet_count", 0)
        category_totals[category]["total_like_count"] += doc.get("like_count", 0)

        # Calculate total engagement and update the dictionary
        total_engagement = (
            category_totals[category]["total_quote_count"]
            + category_totals[category]["total_reply_count"]
            + category_totals[category]["total_retweet_count"]
            + category_totals[category]["total_like_count"]
        )
        category_totals[category]["total_engagement"] = total_engagement

    # Menampilkan hasil
    # for category, totals in category_totals.items():
    #     print(f"Category: {category_totals}")
    #     # print(f"Total Quote Count: {totals['total_quote_count']}")
    #     # print(f"Total Reply Count: {totals['total_reply_count']}")
    #     # print(f"Total Retweet Count: {totals['total_retweet_count']}")
    #     # print(f"Total Like Count: {totals['total_like_count']}")
    #     # print(f"Total Engagement: {totals['total_engagement']}")  # Print total engagement
    #     print()

    return category_totals

def get_posting(target_year, target_month):
    # Koneksi ke MongoDB
    client = pymongo.MongoClient("mongodb+srv://aininurulazizah:aininurulazizah@cluster0.cmotjqx.mongodb.net/?retryWrites=true&w=majority")

    # Memilih database
    db = client.get_database('tweets_db')

    # Mengambil koleksi
    records = db.data_tweets

    # Query untuk mencari data dengan tahun dan bulan tertentu
    date_query = {
        "$expr": {
            "$and": [
                {"$eq": [{"$year": {"$toDate": "$post_date"}}, target_year]},
                {"$eq": [{"$month": {"$toDate": "$post_date"}}, target_month]}
            ]
        }
    }

    # Proyeksi untuk hanya mengambil kolom yang diperlukan
    projection = {"tweet_text": 1, "_id": 0}

    # Membuat pipeline agregasi
    pipeline = [
        {"$match": date_query},
        {"$project": projection}
    ]

    # Melakukan agregasi
    result = list(records.aggregate(pipeline))

    # Simpan data dalam file CSV
    csv_filename = f"twitter_data_{target_year}_{target_month}.csv"
    with open(csv_filename, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = ['tweet_text']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for record in result:
            writer.writerow({'tweet_text': record.get('tweet_text', '')})

    return csv_filename
