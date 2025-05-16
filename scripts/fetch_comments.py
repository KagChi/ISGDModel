from googleapiclient.discovery import build
from dotenv import load_dotenv
import csv
import html
import argparse
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def video_comments_flat(video_id, api_key):
    """
    Fetches all comments (including replies) from a YouTube video.

    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your YouTube API key.

    Returns:
        list: A list of strings, where each string is a comment or a reply.  Returns an empty list if there's an error.
    """
    texts = []
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part='snippet,replies',
            videoId=video_id,
            maxResults=100
        )

        while request:
            response = request.execute()

            for item in response.get('items', []):
                # Top level comment text
                comment_text = html.unescape(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                texts.append(comment_text)

                # Add replies if any
                if item['snippet']['totalReplyCount'] > 0:
                    for reply_item in item.get('replies', {}).get('comments', []):
                        reply_text = html.unescape(reply_item['snippet']['textDisplay'])
                        texts.append(reply_text)

            request = youtube.commentThreads().list_next(request, response)

        logger.info(f"Successfully fetched {len(texts)} comments from video ID: {video_id}")
        return texts

    except Exception as e:
        logger.error(f"An error occurred while fetching comments for video ID {video_id}: {e}", exc_info=True)
        return []


def main():
    """
    Main function to fetch comments from a YouTube video and save them to a CSV file.
    """
    parser = argparse.ArgumentParser(description="Fetch comments from a YouTube video.")
    parser.add_argument("video_id", help="The ID of the YouTube video.")
    # parser.add_argument("--output_file", default="data/comments.csv", help="The output CSV file (default: data/comments.csv).")

    args = parser.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY environment variable not set.")
        print("Error: YOUTUBE_API_KEY environment variable not set.") # Keep the print for immediate user feedback
        return

    all_texts = video_comments_flat(args.video_id, api_key)

    if not all_texts:
        logger.warning(f"No comments fetched for video ID: {args.video_id}.  CSV will be empty.")
        return

    # Generate output file name based on video ID and current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"csv/comments/{timestamp}.csv"

    output_dir = os.path.dirname(output_file)
    if output_dir:  # Check if the output directory is not empty
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['text'])  # single header column
            for text in all_texts:
                writer.writerow([text])
        logger.info(f"Comments saved to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred while writing to the CSV file: {e}", exc_info=True)
        print(f"Error writing to CSV file: {e}") # Keep the print for immediate user feedback


if __name__ == "__main__":
    main()
