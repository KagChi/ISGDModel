from googleapiclient.discovery import build
import csv
import html
import argparse
import os

def video_comments_flat(video_id, api_key):
    texts = []

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

    return texts

def main():
    parser = argparse.ArgumentParser(description="Fetch comments from a YouTube video.")
    parser.add_argument("video_id", help="The ID of the YouTube video.")
    parser.add_argument("--output_file", default="data/comments.csv", help="The output CSV file (default: data/comments.csv).")

    args = parser.parse_args()

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set.")
        return

    all_texts = video_comments_flat(args.video_id, api_key)

    with open(args.output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text'])  # single header column
        for text in all_texts:
            writer.writerow([text])

if __name__ == "__main__":
    main()
