import re
from datetime import datetime
from typing import List, Dict

def parse_timestamp(timestamp: str) -> float:
    """Convert timestamp string to seconds."""
    try:
        time_parts = timestamp.split(':')
        if len(time_parts) == 2:
            minutes, seconds = time_parts
            return int(minutes) * 60 + float(seconds)
        elif len(time_parts) == 3:
            hours, minutes, seconds = time_parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return 0

def parse_gemini_response(response_text: str) -> List[Dict]:
    """Parse Gemini's response into structured clip data."""
    clips = []
    current_clip = {}
    
    # Split response into clip segments
    segments = response_text.split('\n\n')
    
    for segment in segments:
        lines = segment.strip().split('\n')
        
        # Parse timestamp line
        timestamp_match = re.match(r'(\d{1,2}:\d{2}(?::\d{2})?) - (\d{1,2}:\d{2}(?::\d{2})?)', lines[0])
        if timestamp_match:
            start_time, end_time = timestamp_match.groups()
            current_clip = {
                'start_time': parse_timestamp(start_time),
                'end_time': parse_timestamp(end_time),
                'description': '',
                'viral_score': 0,
                'platforms': []
            }
            
            # Parse remaining lines
            for line in lines[1:]:
                if line.startswith('Description:'):
                    current_clip['description'] = line.replace('Description:', '').strip()
                elif line.startswith('Viral Potential:'):
                    try:
                        score = int(re.search(r'\d+', line).group())
                        current_clip['viral_score'] = min(max(score, 1), 10)
                    except (AttributeError, ValueError):
                        current_clip['viral_score'] = 5
                elif line.startswith('Best Platforms:'):
                    platforms = line.replace('Best Platforms:', '').strip()
                    current_clip['platforms'] = [p.strip() for p in platforms.split(',')]
            
            if current_clip['end_time'] > current_clip['start_time']:
                clips.append(current_clip)
    
    return clips