from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import os
import google.generativeai as genai
from yt_dlp import YoutubeDL
import ffmpeg
from dotenv import load_dotenv
import uuid 
import logging
import time
import re
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://localhost:5050", "https://sb1tjeckc-sayr--5173--fc837ba8.local-corp.webcontainer.io", "https://boisterous-treacle-a2525d.netlify.app"],
    # allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_AI_API_KEY"))

# Mount the clips directory to serve files
app.mount("/clips", StaticFiles(directory="clips"), name="clips")

class VideoRequest(BaseModel):
    url: str
    options: Dict = {}

class ProcessingJob:
    def __init__(self):
        self.state = "processing"
        self.clips = []
        self.highlights = None
        self.error = None
        self.message = None

active_jobs = {}

def convert_timestamp_to_seconds(timestamp: str) -> int:
    """Convert MM:SS or HH:MM:SS format to seconds"""
    parts = timestamp.strip().split(':')
    if len(parts) == 2:  # MM:SS
        return int(parts[0]) * 60 + int(parts[1])
    elif len(parts) == 3:  # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    raise ValueError(f"Invalid timestamp format: {timestamp}")

def parse_gemini_response(response_text: str) -> list:
    """Parse Gemini's response into structured clip data"""
    logging.info("Starting to parse Gemini response...")
    clips = []
    current_clip = None
    
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Remove markdown formatting if present
        line = line.replace('**', '')
            
        # Look for timestamp pattern (MM:SS - MM:SS)
        timestamp_match = re.search(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', line)
        if timestamp_match:
            if current_clip:
                clips.append(current_clip)
            
            start_str, end_str = timestamp_match.groups()
            try:
                # Add 1 second to the end time
                end_seconds = convert_timestamp_to_seconds(end_str) + 1
                current_clip = {
                    'start_time': convert_timestamp_to_seconds(start_str),
                    'end_time': end_seconds,
                    'original_end': end_str,  # Keep original for reference
                    'description': '',
                    'viral_potential': 0,
                    'platforms': []
                }
                logging.debug(f"Found timestamp: {start_str} - {end_str} (adjusted end to +1 second)")
            except ValueError as e:
                logging.error(f"Error parsing timestamp: {e}")
                continue
        
        elif current_clip:
            # Handle description (might be on the same line after timestamp)
            if 'Description:' in line:
                current_clip['description'] = line.split('Description:', 1)[1].strip()
                logging.debug(f"Found description: {current_clip['description'][:50]}...")
            
            # Handle viral potential (now handles both "7" and "7/10" formats)
            elif 'Viral Potential:' in line:
                try:
                    potential_str = line.split('Viral Potential:', 1)[1].strip()
                    potential = int(potential_str.split('/')[0])  # Handle both "7" and "7/10"
                    current_clip['viral_potential'] = potential
                    logging.debug(f"Found viral potential: {potential}")
                except (ValueError, IndexError) as e:
                    logging.error(f"Error parsing viral potential: {e}")
                    
            # Handle platforms
            elif 'Best Platforms:' in line:
                platforms_str = line.split('Best Platforms:', 1)[1].strip()
                # Handle both comma-separated and space-separated platforms
                platforms = [p.strip(' ,') for p in re.split(r'[,\s]+', platforms_str) if p.strip(' ,')]
                current_clip['platforms'] = platforms
                logging.debug(f"Found platforms: {platforms}")
    
    # Don't forget to add the last clip
    if current_clip:
        clips.append(current_clip)
    
    logging.info(f"Found {len(clips)} clips in Gemini response")
    if not clips:
        logging.warning("No clips were parsed from the response")
        logging.debug("Response may not be in expected format")
    
    return clips

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename and make it filesystem safe"""
    # Remove or replace invalid characters
    invalid_chars = r'<>:"/\|?*'
    # Replace special characters with underscore
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Replace multiple spaces/underscores with single underscore
    filename = re.sub(r'[\s_]+', '_', filename)
    
    # Remove any non-ASCII characters
    filename = re.sub(r'[^\x00-\x7F]+', '', filename)
    
    # Trim length if needed (Windows has 255 char limit)
    if len(filename) > 200:  # Leave room for additional chars
        filename = filename[:200]
    
    return filename.strip('_')  # Remove leading/trailing underscores

def cleanup_gemini_files():
    """Clean up any lingering files from Gemini"""
    try:
        logging.info("Checking for files to clean up in Gemini...")
        for f in genai.list_files():
            try:
                logging.info(f"Deleting Gemini file: {f.name}")
                genai.delete_file(f.name)
            except Exception as e:
                logging.error(f"Error deleting Gemini file {f.name}: {e}")
    except Exception as e:
        logging.error(f"Error listing Gemini files: {e}")

def create_highlights_reel(video_folder: str, clips: list, min_viral_potential: int = 7):
    """Create a highlights reel from clips with high viral potential"""
    try:
        logging.info(f"Creating highlights reel from clips with viral potential >= {min_viral_potential}")
        
        # Filter clips by viral potential and sort by their order in the video
        highlight_clips = [clip for clip in clips if clip['viral_potential'] >= min_viral_potential]
        highlight_clips.sort(key=lambda x: x['start_time'])
        
        if not highlight_clips:
            logging.info("No clips meet the viral potential threshold")
            return None
            
        highlights_path = os.path.join(video_folder, "highlights.mp4")
        reencoded_clips = []
        
        # First pass: Re-encode all clips
        for i, clip in enumerate(highlight_clips):
            try:
                reencoded_path = clip['url'].replace('.mp4', '_reencoded.mp4')
                logging.info(f"Re-encoding clip {i+1}/{len(highlight_clips)}")
                
                ffmpeg.input(clip['url']) \
                      .output(reencoded_path, 
                             vcodec='libx264', 
                             acodec='aac',
                             video_bitrate='2000k',
                             audio_bitrate='128k') \
                      .overwrite_output() \
                      .run(capture_stdout=True, capture_stderr=True)
                
                reencoded_clips.append(reencoded_path)
                logging.info(f"Successfully re-encoded clip {i+1}")
                
            except Exception as e:
                logging.error(f"Error re-encoding clip {i+1}: {e}")
                continue
        
        if not reencoded_clips:
            raise Exception("No clips were successfully re-encoded")
        
        # Create concat file with re-encoded clips
        concat_file = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file, 'w', encoding='utf-8') as f:
            for clip_path in reencoded_clips:
                # Use absolute paths and escape backslashes for Windows
                abs_path = os.path.abspath(clip_path).replace('\\', '\\\\')
                f.write(f"file '{abs_path}'\n")
        
        logging.info(f"Concatenating {len(reencoded_clips)} clips for highlights reel")
        
        try:
            # Use absolute path for output file too
            abs_highlights_path = os.path.abspath(highlights_path)
            
            # Concatenate using the concat demuxer
            ffmpeg.input(concat_file, format='concat', safe=0) \
                  .output(abs_highlights_path, c='copy') \
                  .overwrite_output() \
                  .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                  
            logging.info(f"Successfully created highlights reel: {highlights_path}")
            
            # Add highlights info to analysis.md
            total_duration = sum(clip['end_time'] - clip['start_time'] for clip in highlight_clips)
            analysis_path = os.path.join(video_folder, "analysis.md")
            
            with open(analysis_path, 'a', encoding='utf-8') as f:
                f.write("\n\n## Highlights Reel\n\n")
                f.write(f"Total Duration: {total_duration} seconds\n\n")
                f.write("Included Clips:\n\n")
                for clip in highlight_clips:
                    f.write(f"- {clip.get('original_end', '')} (Viral Potential: {clip['viral_potential']})\n")
                    f.write(f"  {clip['description']}\n\n")
            
            return highlights_path
            
        except Exception as e:
            logging.error(f"FFmpeg error creating highlights: {str(e)}")
            return None
            
    except Exception as e:
        logging.error(f"Error creating highlights reel: {str(e)}")
        return None
    finally:
        # Cleanup temporary files
        try:
            if 'concat_file' in locals() and os.path.exists(concat_file):
                os.remove(concat_file)
            # Optionally clean up re-encoded clips after successful concatenation
            if 'reencoded_clips' in locals() and os.path.exists(highlights_path):
                for clip_path in reencoded_clips:
                    if os.path.exists(clip_path):
                        os.remove(clip_path)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

def process_video_task(job_id: str, url: str, options: dict):
    """Main video processing task"""
    job = active_jobs[job_id]
    video_path = None
    try:
        # Download video
        job.message = "Starting video download..."
        logging.info(job.message)
        ydl_opts = {
            'format': 'best',
            'quiet': True,
            'no_warnings': True,
            'outtmpl': f'downloads/{job_id}_%(title)s.%(ext)s'
        }
        
        os.makedirs('downloads', exist_ok=True)
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info)
            video_title = sanitize_filename(info.get('title', 'unknown_video'))
            logging.info(f"Video downloaded: {video_path}")

        # Create a dedicated folder for this video's clips
        video_folder = f"clips/{video_title}_{job_id}"
        os.makedirs(video_folder, exist_ok=True)
        
        # Upload to Gemini
        logging.info("Uploading to Gemini...")
        video_file = genai.upload_file(path=video_path)
        
        # Wait for processing
        while video_file.state.name == "PROCESSING":
            job.message = "Waiting for Gemini processing..."
            logging.info(job.message)
            time.sleep(5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name != "ACTIVE":
            raise Exception(f"Video processing failed: {video_file.state.name}")

        # Generate content with Gemini
        job.message = "Sending prompt to Gemini..."
        logging.info(job.message)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """Analyze this video and identify the most engaging moments.
        For each moment, provide the information in EXACTLY this format (see example below):

        MM:SS - MM:SS
        Description: [Describe what happens in this clip]
        Viral Potential: [Rate from 1-10]
        Best Platforms: [List suitable social platforms]

        Example format:
        2:15 - 3:45
        Description: Person explains the key concept with a surprising revelation
        Viral Potential: 8
        Best Platforms: TikTok, Instagram Reels, YouTube Shorts

        Keep clips between 1-5 minutes long. 
        Let them make sense in context of the video.
        Focus on moments that would be engaging on social media.
        Do not use any markdown formatting in your response."""

        logging.info("Sending prompt to Gemini:")
        logging.info(f"Prompt:\n{prompt}")

        response = model.generate_content(
            [video_file, prompt],
            request_options={"timeout": 600}
        )

        logging.info("Received response from Gemini:")
        logging.info(f"Response:\n{response.text}")

        # Save Gemini's response as markdown
        analysis_path = os.path.join(video_folder, "analysis.md")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"# AI Analysis for: {video_title}\n\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Identified Clips\n\n")
            f.write(response.text)
        
        # Parse clips from response
        clips = parse_gemini_response(response.text)
        if not clips:
            raise Exception("No valid clips identified")
        
        # Process each clip
        processed_clips = []
        for i, clip in enumerate(clips):
            try:
                job.message = f"Processing clip {i+1}/{len(clips)}"
                logging.info(job.message)
                # Create clip filename using timestamp
                clip_filename = f"clip_{i+1}_{clip['start_time']}_{clip['end_time']}.mp4"
                output_path = os.path.join(video_folder, clip_filename)
                
                duration = clip['end_time'] - clip['start_time']
                
                # Use FFmpeg to create clip
                ffmpeg.input(video_path, ss=clip['start_time'], t=duration) \
                      .output(output_path, acodec='copy', vcodec='copy') \
                      .overwrite_output() \
                      .run(capture_stdout=True, capture_stderr=True)
                
                clip['url'] = output_path
                processed_clips.append(clip)
                logging.info(f"Processed clip {i+1}/{len(clips)}")
                
            except Exception as e:
                logging.error(f"Error processing clip {i}: {e}")
                continue
        
        if not processed_clips:
            raise Exception("Failed to process any clips")
        
        # Create highlights reel
        job.message = "Creating highlights reel..."
        highlights_path = create_highlights_reel(video_folder, processed_clips)
        
        # Update clip URLs to use the /clips endpoint
        processed_clips = []
        for clip in clips:
            relative_path = os.path.relpath(clip['url'], 'clips').replace('\\', '/')
            processed_clips.append({
                **clip,
                'url': f"/clips/{relative_path}"
            })

        job.clips = processed_clips

        # Handle highlights if they exist
        if highlights_path and os.path.exists(highlights_path):
            relative_highlights = os.path.relpath(highlights_path, 'clips').replace('\\', '/')
            job.highlights = f"/clips/{relative_highlights}"
            logging.info(f"Highlights created at: {job.highlights}")
        else:
            job.highlights = None
            logging.info("No highlights generated")

        # Set completion state
        job.state = "completed"
        job.message = "Processing completed successfully"
        logging.info("Video processing completed successfully")
        
    except Exception as e:
        job.state = "failed"
        job.error = str(e)
        job.message = f"Processing failed: {str(e)}"
        logging.error(f"Processing error: {e}")
        
    finally:
        # Cleanup code
        try:
            job.message = "Cleaning up files..."
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                logging.info(f"Deleted local video file: {video_path}")
        except Exception as e:
            logging.error(f"Cleanup error: {e}")

@app.post("/process-video")
async def process_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """Endpoint to start video processing"""
    job_id = str(uuid.uuid4())
    active_jobs[job_id] = ProcessingJob()
    
    background_tasks.add_task(
        process_video_task,
        job_id,
        request.url,
        request.options
    )
    
    return {"jobId": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Endpoint to check processing status"""
    job = active_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "state": job.state,
        "message": job.message,
        "clips": job.clips if job.state == "completed" else [],
        "highlights": job.highlights if job.state == "completed" else None,
        "error": job.error if job.state == "failed" else None
    }

# Add an endpoint to serve video files
@app.get("/clips/{file_path:path}")
async def get_video(file_path: str):
    video_path = os.path.join("clips", file_path)
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        video_path,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={os.path.basename(file_path)}",
            "Cross-Origin-Resource-Policy": "cross-origin",
            "Cross-Origin-Embedder-Policy": "require-corp",
            "Cross-Origin-Opener-Policy": "same-origin"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)