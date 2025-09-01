import base64, os, json
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
from dotenv import load_dotenv
import io

load_dotenv()

# Decode JSON from base64
creds_json = base64.b64decode(os.environ["GOOGLE_CREDENTIALS_JSON_B64"]).decode()
creds_info = json.loads(creds_json)

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID")

credentials = service_account.Credentials.from_service_account_info(
    creds_info, scopes=SCOPES
)
drive_service = build("drive", "v3", credentials=credentials)


def list_pdf_files_in_folder(folder_id):
    """List all PDF files in a specific Google Drive folder."""
    pdf_files = []
    query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    results = drive_service.files().list(
        q=query, fields="files(id, name)"
    ).execute()

    for file in results.get("files", []):
        pdf_files.append((file["id"], file["name"]))
    return pdf_files

def download_pdf(file_id, dest_path):
    """Download a PDF file from Google Drive by ID."""
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(dest_path, "wb")
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    fh.close()
