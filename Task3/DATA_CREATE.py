#!/usr/bin/env python3
import os
import argparse
from icrawler.builtin import GoogleImageCrawler

def download_images_by_era(base_dir, images_per_query):
    # Define eras with a list of search queries for each era.
    eras = {
        '1900s': [
            'vintage 1900s historical photographs',
            '1900s old photograph',
            'early 1900s historical image'
        ],
        '1950s': [
            'vintage 1950s historical photographs',
            '1950s old photograph',
            'mid 1950s historical image'
        ],
        '1970s': [
            'vintage 1970s historical photographs',
            '1970s old photograph',
            'late 1970s historical image'
        ]
    }
    
    # Create the base directory if it doesn't exist.
    os.makedirs(base_dir, exist_ok=True)
    
    # For each era, iterate through the list of queries.
    for era, queries in eras.items():
        era_dir = os.path.join(base_dir, era)
        os.makedirs(era_dir, exist_ok=True)
        for idx, query in enumerate(queries):
            # Create a subfolder for each query to keep images organized.
            query_folder = os.path.join(era_dir, f"query_{idx+1}")
            os.makedirs(query_folder, exist_ok=True)
            print(f"Downloading {images_per_query} images for era '{era}' with query: '{query}' into folder: {query_folder}")
            try:
                crawler = GoogleImageCrawler(storage={'root_dir': query_folder})
                crawler.crawl(keyword=query, max_num=images_per_query)
                print(f"Downloaded images for era '{era}', query '{query}' to {query_folder}\n")
            except Exception as e:
                print(f"Error downloading images for era '{era}', query '{query}': {e}")
    
    print("Download complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Download a large and relevant dataset of historical images by era using icrawler."
    )
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='dataset/historical_photos_by_era',
        help='Base directory to save images (default: dataset/historical_photos_by_era)'
    )
    parser.add_argument(
        '--images_per_query', 
        type=int, 
        default=1000,
        help='Number of images to download per query (default: 1000)'
    )
    args = parser.parse_args()
    
    download_images_by_era(args.base_dir, args.images_per_query)

if __name__ == '__main__':
    main()
