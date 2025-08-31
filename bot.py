import os
import re
import json
import time
import logging
import schedule
import requests
import random
from datetime import datetime, timedelta
import yt_dlp
from pathlib import Path
import hashlib
import shutil
import threading
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instagram_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """Decorator for retrying functions with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    wait_time = backoff_factor ** attempt + random.uniform(1, 3)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

class HumanLikeDelays:
    """Class to simulate human-like behavior with realistic delays"""
    
    @staticmethod
    def reading_delay(text_length=100):
        """Simulate time to read text (average 200 words per minute)"""
        words = text_length / 5  # Average 5 characters per word
        read_time = (words / 200) * 60  # 200 WPM converted to seconds
        actual_delay = max(2, read_time + random.uniform(1, 3))
        time.sleep(actual_delay)
    
    @staticmethod
    def typing_delay(text_length=50):
        """Simulate typing time (average 40 WPM)"""
        words = text_length / 5
        type_time = (words / 40) * 60  # 40 WPM for typing
        actual_delay = max(1, type_time + random.uniform(0.5, 2))
        time.sleep(actual_delay)
    
    @staticmethod
    def browsing_delay():
        """Simulate browsing between actions"""
        time.sleep(random.uniform(3, 8))
    
    @staticmethod
    def network_delay():
        """Simulate network latency"""
        time.sleep(random.uniform(0.5, 2))
    
    @staticmethod
    def processing_delay():
        """Simulate processing/thinking time"""
        time.sleep(random.uniform(2, 5))

class SessionManager:
    """Manages Instagram login session persistence"""
    
    def __init__(self, session_file="session.json"):
        self.session_file = session_file
        self.session_data = self.load_session()
        self.last_refresh = datetime.now()
    
    def load_session(self):
        """Load session data from file"""
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = json.load(f)
                    logger.info("Loaded existing session data")
                    return data
        except Exception as e:
            logger.error(f"Error loading session: {e}")
        return {}
    
    def save_session(self):
        """Save session data to file"""
        try:
            # Create backup first
            if os.path.exists(self.session_file):
                shutil.copy2(self.session_file, f"{self.session_file}.backup")
            
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
                logger.info("Session data saved")
        except Exception as e:
            logger.error(f"Error saving session: {e}")
    
    def update_tokens(self, access_token, expires_at=None):
        """Update access token and expiry"""
        self.session_data['access_token'] = access_token
        if expires_at:
            self.session_data['expires_at'] = expires_at
        self.session_data['last_updated'] = datetime.now().isoformat()
        self.save_session()
    
    def is_token_valid(self):
        """Check if current token is still valid"""
        if 'expires_at' not in self.session_data:
            return True  # Assume valid if no expiry set
        
        try:
            expiry = datetime.fromisoformat(self.session_data['expires_at'])
            return datetime.now() < expiry - timedelta(hours=1)  # Refresh 1 hour before expiry
        except:
            return False
    
    def get_access_token(self):
        """Get current access token"""
        return self.session_data.get('access_token') or os.getenv('INSTAGRAM_ACCESS_TOKEN')

class BackupManager:
    """Manages backups and data recovery"""
    
    def __init__(self, backup_dir="backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, file_path, description=""):
        """Create timestamped backup of a file"""
        try:
            if not os.path.exists(file_path):
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(file_path).name
            backup_name = f"{timestamp}_{filename}"
            if description:
                backup_name = f"{timestamp}_{description}_{filename}"
            
            backup_path = self.backup_dir / backup_name
            shutil.copy2(file_path, backup_path)
            
            logger.info(f"Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def restore_latest_backup(self, original_file):
        """Restore the latest backup of a file"""
        try:
            filename = Path(original_file).name
            backups = list(self.backup_dir.glob(f"*{filename}"))
            
            if not backups:
                logger.warning(f"No backups found for {filename}")
                return False
            
            # Sort by modification time, get latest
            latest_backup = max(backups, key=os.path.getmtime)
            shutil.copy2(latest_backup, original_file)
            
            logger.info(f"Restored from backup: {latest_backup}")
            return True
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def cleanup_old_backups(self, days=7):
        """Remove backups older than specified days"""
        try:
            cutoff = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for backup_file in self.backup_dir.glob("*"):
                if backup_file.stat().st_mtime < cutoff.timestamp():
                    backup_file.unlink()
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old backups")
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")

class InstagramReelsBot:
    def __init__(self):
        # Initialize managers
        self.session_manager = SessionManager()
        self.backup_manager = BackupManager()
        self.delays = HumanLikeDelays()
        
        # Instagram API credentials
        self.access_token = self.session_manager.get_access_token()
        self.page_id = os.getenv('INSTAGRAM_PAGE_ID')
        self.app_id = os.getenv('INSTAGRAM_APP_ID')
        self.app_secret = os.getenv('INSTAGRAM_APP_SECRET')
        
        # API endpoints
        self.graph_api_base = "https://graph.facebook.com/v18.0"
        
        # Storage for processed posts
        self.processed_posts_file = "processed_posts.json"
        self.processed_posts = self.load_processed_posts()
        
        # Download directory
        self.download_dir = Path("downloads")
        self.download_dir.mkdir(exist_ok=True)
        
        # Rate limiting
        self.last_api_call = 0
        self.api_call_interval = 2  # Minimum seconds between API calls
        
        # Health check
        self.last_successful_run = datetime.now()
        
        if not all([self.access_token, self.page_id]):
            raise ValueError("Missing required credentials")
    
    def load_processed_posts(self):
        """Load list of already processed post URLs with backup recovery"""
        try:
            if os.path.exists(self.processed_posts_file):
                with open(self.processed_posts_file, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data)} processed posts")
                    return data
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error loading processed posts: {e}")
            
            # Try to restore from backup
            if self.backup_manager.restore_latest_backup(self.processed_posts_file):
                return self.load_processed_posts()  # Recursive call after restore
        
        logger.info("Starting with empty processed posts list")
        return []
    
    def save_processed_posts(self):
        """Save list of processed post URLs with backup"""
        try:
            # Create backup first
            self.backup_manager.create_backup(self.processed_posts_file, "processed_posts")
            
            with open(self.processed_posts_file, 'w') as f:
                json.dump(self.processed_posts, f, indent=2)
                logger.info(f"Saved {len(self.processed_posts)} processed posts")
        except Exception as e:
            logger.error(f"Error saving processed posts: {e}")
    
    @retry_with_backoff(max_retries=3)
    def rate_limited_request(self, method, url, **kwargs):
        """Make rate-limited API requests"""
        # Ensure minimum interval between requests
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.api_call_interval:
            wait_time = self.api_call_interval - time_since_last
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        # Add human-like network delay
        self.delays.network_delay()
        
        response = requests.request(method, url, **kwargs)
        self.last_api_call = time.time()
        
        # Check for rate limiting
        if response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            logger.warning(f"Rate limited. Waiting {retry_after}s")
            time.sleep(retry_after + random.uniform(5, 15))
            raise requests.exceptions.RequestException("Rate limited")
        
        response.raise_for_status()
        return response
    
    @retry_with_backoff(max_retries=3)
    def refresh_access_token(self):
        """Refresh access token if needed"""
        if self.session_manager.is_token_valid():
            return True
        
        try:
            logger.info("Refreshing access token...")
            
            url = f"{self.graph_api_base}/oauth/access_token"
            params = {
                'grant_type': 'fb_exchange_token',
                'client_id': self.app_id,
                'client_secret': self.app_secret,
                'fb_exchange_token': self.access_token
            }
            
            response = self.rate_limited_request('GET', url, params=params)
            data = response.json()
            
            new_token = data.get('access_token')
            expires_in = data.get('expires_in', 3600)
            expires_at = (datetime.now() + timedelta(seconds=expires_in)).isoformat()
            
            if new_token:
                self.access_token = new_token
                self.session_manager.update_tokens(new_token, expires_at)
                logger.info("Access token refreshed successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to refresh access token: {e}")
        
        return False
    
    @retry_with_backoff(max_retries=3)
    def get_instagram_conversations(self):
        """Get Instagram conversations/DMs with human-like behavior"""
        if not self.refresh_access_token():
            return None
        
        logger.info("Checking Instagram conversations...")
        self.delays.browsing_delay()
        
        url = f"{self.graph_api_base}/{self.page_id}/conversations"
        params = {
            'platform': 'instagram',
            'access_token': self.access_token,
            'limit': 25  # Reasonable limit
        }
        
        response = self.rate_limited_request('GET', url, params=params)
        data = response.json()
        
        conversations = data.get('data', [])
        logger.info(f"Found {len(conversations)} conversations")
        
        # Simulate reading through conversations
        self.delays.processing_delay()
        
        return data
    
    @retry_with_backoff(max_retries=3)
    def get_conversation_messages(self, conversation_id, since_time=None):
        """Get messages from a specific conversation with human delays"""
        logger.debug(f"Checking messages in conversation: {conversation_id}")
        
        # Simulate opening conversation
        self.delays.browsing_delay()
        
        url = f"{self.graph_api_base}/{conversation_id}/messages"
        params = {
            'access_token': self.access_token,
            'fields': 'id,created_time,from,to,message,attachments',
            'limit': 50
        }
        
        if since_time:
            params['since'] = since_time.isoformat()
        
        response = self.rate_limited_request('GET', url, params=params)
        data = response.json()
        
        messages = data.get('data', [])
        logger.debug(f"Found {len(messages)} recent messages")
        
        # Simulate reading messages
        if messages:
            total_text_length = sum(len(msg.get('message', '')) for msg in messages)
            self.delays.reading_delay(total_text_length)
        
        return data
    
    def extract_instagram_reel_urls(self, text):
        """Extract Instagram reel URLs from message text"""
        if not text:
            return []
        
        # Enhanced patterns to match various Instagram URL formats
        patterns = [
            r'https?://(?:www\.)?instagram\.com/reel/([A-Za-z0-9_-]+)/?(?:\?[^\s]*)?',
            r'https?://(?:www\.)?instagram\.com/p/([A-Za-z0-9_-]+)/?(?:\?[^\s]*)?',
            r'https?://(?:www\.)?instagram\.com/tv/([A-Za-z0-9_-]+)/?(?:\?[^\s]*)?',
            r'https?://(?:www\.)?instagram\.com/stories/[^/]+/([A-Za-z0-9_-]+)/?(?:\?[^\s]*)?'
        ]
        
        urls = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize URL
                if '/reel/' in text or '/p/' in text:
                    full_url = f"https://www.instagram.com/reel/{match}/"
                else:
                    full_url = f"https://www.instagram.com/p/{match}/"
                
                # Check if already processed
                url_hash = hashlib.md5(full_url.encode()).hexdigest()
                if url_hash not in [hashlib.md5(url.encode()).hexdigest() for url in self.processed_posts]:
                    urls.append(full_url)
                    logger.info(f"Found new reel: {full_url}")
        
        return urls
    
    @retry_with_backoff(max_retries=3)
    def download_reel(self, url):
        """Download Instagram reel using yt-dlp with multiple fallbacks"""
        logger.info(f"Downloading reel: {url}")
        
        # Simulate deciding to download
        self.delays.processing_delay()
        
        # Try different yt-dlp configurations
        configurations = [
            {
                'outtmpl': str(self.download_dir / '%(title)s_%(id)s.%(ext)s'),
                'format': 'best[height<=1080][ext=mp4]',
                'writeinfojson': True,
                'no_warnings': True,
            },
            {
                'outtmpl': str(self.download_dir / '%(id)s.%(ext)s'),
                'format': 'best[ext=mp4]',
                'writeinfojson': True,
                'no_warnings': True,
            },
            {
                'outtmpl': str(self.download_dir / '%(id)s.%(ext)s'),
                'format': 'best',
                'writeinfojson': True,
                'no_warnings': True,
            }
        ]
        
        for i, ydl_opts in enumerate(configurations):
            try:
                logger.info(f"Trying download configuration {i+1}")
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract info first
                    info = ydl.extract_info(url, download=False)
                    
                    if info.get('_type') == 'playlist':
                        logger.warning(f"Skipping playlist: {url}")
                        continue
                    
                    # Check duration (skip if too long)
                    duration = info.get('duration', 0)
                    if duration > 300:  # 5 minutes max
                        logger.warning(f"Video too long ({duration}s), skipping: {url}")
                        continue
                    
                    # Download with progress simulation
                    logger.info("Starting download...")
                    ydl.download([url])
                    
                    # Find downloaded file
                    video_id = info.get('id')
                    title = info.get('title', 'Unknown').replace('/', '_')[:50]  # Sanitize title
                    
                    for file_path in self.download_dir.glob(f"*{video_id}*"):
                        if file_path.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
                            logger.info(f"Successfully downloaded: {file_path}")
                            return {
                                'file_path': file_path,
                                'title': title,
                                'description': info.get('description', '')[:200],  # Limit description
                                'original_url': url,
                                'duration': duration
                            }
                    
                    logger.warning(f"Downloaded file not found for {url}")
                    
            except Exception as e:
                logger.warning(f"Download configuration {i+1} failed: {e}")
                if i == len(configurations) - 1:  # Last attempt
                    raise
                continue
        
        return None
    
    @retry_with_backoff(max_retries=3)
    def upload_reel_to_instagram(self, video_info):
        """Upload downloaded reel to Instagram with human-like behavior"""
        logger.info(f"Preparing to upload: {video_info['title']}")
        
        # Simulate reviewing the content
        self.delays.processing_delay()
        
        video_path = video_info['file_path']
        
        # Prepare caption with human-like writing delay
        caption_parts = [
            video_info['title'],
            "",
            "📱 Reposted",
            "#reels #viral #content"
        ]
        
        caption = "\n".join(caption_parts)
        if len(caption) > 2200:
            caption = caption[:2200] + "..."
        
        # Simulate typing caption
        self.delays.typing_delay(len(caption))
        
        try:
            # Step 1: Create media container
            logger.info("Creating media container...")
            
            container_url = f"{self.graph_api_base}/{self.page_id}/media"
            
            with open(video_path, 'rb') as video_file:
                files = {'source': video_file}
                data = {
                    'media_type': 'REELS',
                    'caption': caption,
                    'access_token': self.access_token
                }
                
                # Simulate upload process
                logger.info("Uploading video...")
                response = self.rate_limited_request('POST', container_url, data=data, files=files)
                container_data = response.json()
            
            container_id = container_data.get('id')
            if not container_id:
                logger.error("Failed to create media container")
                return False
            
            logger.info(f"Media container created: {container_id}")
            
            # Step 2: Wait for processing with human-like patience
            logger.info("Waiting for video processing...")
            if not self.wait_for_processing(container_id):
                return False
            
            # Step 3: Publish with final delay
            logger.info("Publishing reel...")
            self.delays.processing_delay()
            
            publish_url = f"{self.graph_api_base}/{self.page_id}/media_publish"
            publish_data = {
                'creation_id': container_id,
                'access_token': self.access_token
            }
            
            response = self.rate_limited_request('POST', publish_url, data=publish_data)
            publish_data = response.json()
            
            post_id = publish_data.get('id')
            if post_id:
                logger.info(f"✅ Successfully posted reel! Post ID: {post_id}")
                return True
            else:
                logger.error("Post creation returned no ID")
                return False
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return False
    
    def wait_for_processing(self, container_id, max_wait=600):
        """Wait for video processing with human-like patience"""
        start_time = time.time()
        check_interval = 15  # Start with 15 second intervals
        
        while time.time() - start_time < max_wait:
            try:
                url = f"{self.graph_api_base}/{container_id}"
                params = {
                    'fields': 'status_code',
                    'access_token': self.access_token
                }
                
                response = self.rate_limited_request('GET', url, params=params)
                data = response.json()
                
                status = data.get('status_code')
                elapsed = int(time.time() - start_time)
                
                if status == 'FINISHED':
                    logger.info(f"✅ Video processing completed in {elapsed}s")
                    return True
                elif status == 'ERROR':
                    logger.error("❌ Video processing failed")
                    return False
                
                logger.info(f"🔄 Processing status: {status} ({elapsed}s elapsed)")
                
                # Human-like waiting - increase interval over time
                wait_time = min(check_interval + random.uniform(5, 15), 60)
                time.sleep(wait_time)
                check_interval = min(check_interval + 5, 60)  # Gradually increase
                
            except Exception as e:
                logger.error(f"Error checking processing status: {e}")
                time.sleep(30)
        
        logger.error(f"⏰ Video processing timeout after {max_wait}s")
        return False
    
    def cleanup_downloaded_files(self):
        """Clean up downloaded files with backup option"""
        try:
            files_cleaned = 0
            for file_path in self.download_dir.glob("*"):
                if file_path.is_file():
                    # Keep some files as backup before cleanup
                    if file_path.stat().st_size > 0:  # Don't backup empty files
                        self.backup_manager.create_backup(str(file_path), "pre_cleanup")
                    file_path.unlink()
                    files_cleaned += 1
            
            if files_cleaned > 0:
                logger.info(f"🧹 Cleaned up {files_cleaned} downloaded files")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def health_check(self):
        """Perform health check and recovery"""
        logger.info("🏥 Performing health check...")
        
        issues_found = []
        
        # Check file system health
        if not os.path.exists(self.processed_posts_file):
            issues_found.append("processed_posts.json missing")
            self.backup_manager.restore_latest_backup(self.processed_posts_file)
        
        # Check API access
        try:
            self.refresh_access_token()
            logger.info("✅ API access healthy")
        except Exception as e:
            issues_found.append(f"API access issue: {e}")
        
        # Check disk space
        try:
            stat = shutil.disk_usage('.')
            free_gb = stat.free / (1024**3)
            if free_gb < 1:  # Less than 1GB free
                issues_found.append(f"Low disk space: {free_gb:.1f}GB")
        except:
            pass
        
        # Update last successful run
        if not issues_found:
            self.last_successful_run = datetime.now()
            logger.info("✅ Health check passed")
        else:
            logger.warning(f"⚠️ Health issues found: {', '.join(issues_found)}")
        
        return len(issues_found) == 0
    
    def process_new_reels(self):
        """Main function to process new reels with comprehensive error handling"""
        logger.info("🚀 Starting reel processing session...")
        
        try:
            # Health check first
            if not self.health_check():
                logger.warning("Health check failed, proceeding with caution...")
            
            # Get conversations with human-like browsing
            conversations = self.get_instagram_conversations()
            if not conversations:
                logger.warning("No conversations found")
                return
            
            # Calculate time threshold
            since_time = datetime.now() - timedelta(hours=25)
            logger.info(f"Looking for reels since: {since_time}")
            
            reels_processed = 0
            reels_found = 0
            
            # Process conversations like a human would
            for i, conversation in enumerate(conversations.get('data', [])):
                conversation_id = conversation.get('id')
                if not conversation_id:
                    continue
                
                logger.info(f"📱 Checking conversation {i+1}/{len(conversations['data'])}")
                
                # Get recent messages
                messages = self.get_conversation_messages(conversation_id, since_time)
                if not messages:
                    continue
                
                # Process messages
                for message in messages.get('data', []):
                    message_text = message.get('message', '')
                    created_time = message.get('created_time')
                    
                    if message_text:
                        logger.debug(f"Message from {created_time}: {message_text[:50]}...")
                    
                    # Extract reel URLs
                    reel_urls = self.extract_instagram_reel_urls(message_text)
                    reels_found += len(reel_urls)
                    
                    for url in reel_urls:
                        logger.info(f"🎬 Processing reel {reels_processed + 1}: {url}")
                        
                        try:
                            # Download with retries
                            video_info = self.download_reel(url)
                            if not video_info:
                                logger.warning(f"❌ Failed to download: {url}")
                                continue
                            
                            # Human-like review pause
                            logger.info("👀 Reviewing downloaded content...")
                            self.delays.processing_delay()
                            
                            # Upload to Instagram
                            if self.upload_reel_to_instagram(video_info):
                                self.processed_posts.append(url)
                                reels_processed += 1
                                logger.info(f"✅ Successfully reposted #{reels_processed}: {video_info['title']}")
                                
                                # Save progress immediately
                                self.save_processed_posts()
                            else:
                                logger.error(f"❌ Failed to repost: {url}")
                        
                        except Exception as e:
                            logger.error(f"❌ Error processing {url}: {e}")
                        
                        finally:
                            # Clean up individual file
                            if 'video_info' in locals() and video_info:
                                try:
                                    video_info['file_path'].unlink()
                                    logger.debug("🗑️ Cleaned up download file")
                                except:
                                    pass
                        
                        # Human-like delay between posts
                        if reels_processed > 0:
                            wait_time = random.uniform(45, 90)  # 45-90 seconds between posts
                            logger.info(f"😴 Resting for {wait_time:.0f}s before next post...")
                            time.sleep(wait_time)
                
                # Delay between conversations
                if i < len(conversations['data']) - 1:
                    self.delays.browsing_delay()
            
            # Final cleanup and summary
            self.cleanup_downloaded_files()
            self.backup_manager.cleanup_old_backups()
            
            logger.info(f"🎯 Session complete! Found: {reels_found} reels, Successfully posted: {reels_processed}")
            
            # Update health status
            self.last_successful_run = datetime.now()
            
        except Exception as e:
            logger.error(f"💥 Critical error in processing session: {e}")
            
            # Emergency cleanup
            try:
                self.cleanup_downloaded_files()
            except:
                pass
    
    def run_scheduled(self):
        """Run the bot on a schedule with monitoring"""
        logger.info("🤖 Instagram Reels Bot started with human-like behavior")
        
        # Clean up old backups on startup
        self.backup_manager.cleanup_old_backups()
        
        # Schedule daily runs at human-like times
        schedule.every().day.at("09:00").do(self.process_new_reels)
        schedule.every().day.at("15:00").do(self.process_new_reels)  # Optional second check
        
        # Health check every 6 hours
        schedule.every(6).hours.do(self.health_check)
        
        # Monitor for stuck processes
        schedule.every().hour.do(self.monitor_health)
        
        # Run once immediately for testing (commented out for production)
        # self.process_new_reels()
        
        logger.info("📅 Scheduled tasks:")
        logger.info("  - Main processing: 9:00 AM and 3:00 PM daily")
        logger.info("  - Health checks: Every 6 hours")
        logger.info("  - Monitoring: Every hour")
        
        # Main loop with graceful error handling
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while True:
            try:
                schedule.run_pending()
                consecutive_errors = 0  # Reset on success
                
                # Sleep with random variation to appear more human
                sleep_time = 3600 + random.uniform(-300, 300)  # 1 hour ± 5 minutes
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"💥 Scheduler error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.critical("🚨 Too many consecutive errors, attempting recovery...")
                    self.emergency_recovery()
                    consecutive_errors = 0
                
                # Exponential backoff for errors
                wait_time = min(300, 30 * (2 ** consecutive_errors))
                logger.info(f"⏳ Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    def monitor_health(self):
        """Monitor bot health and detect issues"""
        current_time = datetime.now()
        
        # Check if last successful run was too long ago
        time_since_success = current_time - self.last_successful_run
        if time_since_success > timedelta(hours=26):  # More than 26 hours
            logger.warning(f"⚠️ No successful run for {time_since_success}")
            
            # Attempt self-recovery
            logger.info("🔧 Attempting self-recovery...")
            try:
                self.emergency_recovery()
            except Exception as e:
                logger.error(f"Self-recovery failed: {e}")
        
        # Check file system
        try:
            # Verify critical files exist
            if not os.path.exists(self.processed_posts_file):
                logger.warning("📄 Processed posts file missing, restoring...")
                self.backup_manager.restore_latest_backup(self.processed_posts_file)
                self.processed_posts = self.load_processed_posts()
        except Exception as e:
            logger.error(f"File system check failed: {e}")
    
    def emergency_recovery(self):
        """Emergency recovery procedures"""
        logger.info("🚨 Starting emergency recovery...")
        
        try:
            # 1. Restore critical files from backup
            self.backup_manager.restore_latest_backup(self.processed_posts_file)
            self.processed_posts = self.load_processed_posts()
            
            # 2. Clean up any corrupted downloads
            self.cleanup_downloaded_files()
            
            # 3. Refresh authentication
            self.refresh_access_token()
            
            # 4. Test API connectivity
            test_response = self.get_instagram_conversations()
            if test_response:
                logger.info("✅ Emergency recovery successful")
                self.last_successful_run = datetime.now()
            else:
                logger.error("❌ Emergency recovery failed - API test failed")
            
        except Exception as e:
            logger.error(f"💥 Emergency recovery failed: {e}")

class SafeFileHandler:
    """Handle file operations with atomic writes and backups"""
    
    @staticmethod
    def safe_write_json(file_path, data):
        """Safely write JSON data with atomic operation"""
        temp_path = f"{file_path}.tmp"
        backup_path = f"{file_path}.backup"
        
        try:
            # Create backup if original exists
            if os.path.exists(file_path):
                shutil.copy2(file_path, backup_path)
            
            # Write to temp file first
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic move
            shutil.move(temp_path, file_path)
            
            # Clean up backup after successful write
            if os.path.exists(backup_path):
                os.remove(backup_path)
            
            return True
        except Exception as e:
            logger.error(f"Safe write failed: {e}")
            
            # Restore from backup if available
            if os.path.exists(backup_path):
                shutil.move(backup_path, file_path)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return False

def setup_github_environment():
    """Setup function for GitHub Actions environment"""
    logger.info("🐙 Setting up GitHub Actions environment...")
    
    # Install additional dependencies if needed
    required_packages = ['schedule', 'yt-dlp', 'requests']
    
    try:
        import pkg_resources
        installed_packages = [d.project_name.lower() for d in pkg_resources.working_set]
        
        missing_packages = [pkg for pkg in required_packages if pkg.lower() not in installed_packages]
        
        if missing_packages:
            import subprocess
            import sys
            
            logger.info(f"📦 Installing missing packages: {missing_packages}")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        
    except Exception as e:
        logger.warning(f"Package check failed: {e}")
    
    # Verify environment variables
    required_env_vars = [
        'INSTAGRAM_ACCESS_TOKEN',
        'INSTAGRAM_PAGE_ID',
        'INSTAGRAM_APP_ID',
        'INSTAGRAM_APP_SECRET'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    logger.info("✅ GitHub Actions environment ready")
    return True

def run_single_execution():
    """Run a single execution (for GitHub Actions)"""
    logger.info("🎯 Running single execution mode...")
    
    if not setup_github_environment():
        logger.error("❌ Environment setup failed")
        return
    
    try:
        bot = InstagramReelsBot()
        bot.process_new_reels()
        logger.info("✅ Single execution completed successfully")
    except Exception as e:
        logger.error(f"💥 Single execution failed: {e}")
        raise

def main():
    """Main function with execution mode detection"""
    # Detect if running in GitHub Actions
    if os.getenv('GITHUB_ACTIONS') or os.getenv('RUNNER_OS'):
        run_single_execution()
    else:
        # Local continuous mode
        try:
            if not setup_github_environment():
                logger.error("❌ Environment setup failed")
                return
            
            bot = InstagramReelsBot()
            bot.run_scheduled()
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped by user")
        except Exception as e:
            logger.error(f"💥 Bot crashed: {e}")
            
            # Attempt emergency recovery
            try:
                bot = InstagramReelsBot()
                bot.emergency_recovery()
            except:
                logger.error("🚨 Emergency recovery also failed")

if __name__ == "__main__":
    main()