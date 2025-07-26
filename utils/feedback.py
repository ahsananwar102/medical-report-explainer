"""
Feedback utility module for storing and managing user feedback
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import streamlit as st

from config.config import FEEDBACK_FILE


class FeedbackManager:
    """Manage user feedback storage and retrieval"""
    
    def __init__(self, feedback_file: Path = FEEDBACK_FILE):
        """Initialize feedback manager"""
        self.feedback_file = feedback_file
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist
        if not self.feedback_file.exists():
            self._initialize_feedback_file()
    
    def _initialize_feedback_file(self):
        """Create initial feedback file structure"""
        initial_data = {
            "feedback_entries": [],
            "created_at": datetime.now().isoformat(),
            "total_feedback": 0
        }
        
        with open(self.feedback_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
    
    def add_feedback(self, feedback_type: str, rating: str, 
                    query: str = "", response: str = "", 
                    comments: str = "") -> bool:
        """
        Add user feedback
        
        Args:
            feedback_type: Type of feedback ('explanation', 'summary', 'question')
            rating: User rating ('thumbs_up', 'thumbs_down')
            query: Original user query
            response: System response
            comments: Optional user comments
            
        Returns:
            Success status
        """
        try:
            # Load existing feedback
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            # Create new feedback entry
            new_feedback = {
                "id": len(feedback_data["feedback_entries"]) + 1,
                "timestamp": datetime.now().isoformat(),
                "feedback_type": feedback_type,
                "rating": rating,
                "query": query,
                "response": response,
                "comments": comments,
                "session_id": st.session_state.get("session_id", "unknown")
            }
            
            # Add to feedback list
            feedback_data["feedback_entries"].append(new_feedback)
            feedback_data["total_feedback"] = len(feedback_data["feedback_entries"])
            feedback_data["last_updated"] = datetime.now().isoformat()
            
            # Save updated feedback
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error saving feedback: {e}")
            return False
    
    def get_feedback_stats(self) -> Dict:
        """
        Get feedback statistics
        
        Returns:
            Dictionary with feedback statistics
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            entries = feedback_data.get("feedback_entries", [])
            
            if not entries:
                return {
                    "total_feedback": 0,
                    "thumbs_up": 0,
                    "thumbs_down": 0,
                    "satisfaction_rate": 0.0,
                    "feedback_by_type": {}
                }
            
            # Calculate statistics
            thumbs_up = sum(1 for entry in entries if entry["rating"] == "thumbs_up")
            thumbs_down = sum(1 for entry in entries if entry["rating"] == "thumbs_down")
            
            satisfaction_rate = (thumbs_up / len(entries)) * 100 if entries else 0
            
            # Feedback by type
            feedback_by_type = {}
            for entry in entries:
                ftype = entry["feedback_type"]
                if ftype not in feedback_by_type:
                    feedback_by_type[ftype] = {"thumbs_up": 0, "thumbs_down": 0}
                feedback_by_type[ftype][entry["rating"]] += 1
            
            return {
                "total_feedback": len(entries),
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "satisfaction_rate": satisfaction_rate,
                "feedback_by_type": feedback_by_type
            }
            
        except Exception as e:
            st.error(f"Error loading feedback stats: {e}")
            return {}
    
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """
        Get recent feedback entries
        
        Args:
            limit: Number of recent entries to return
            
        Returns:
            List of recent feedback entries
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            entries = feedback_data.get("feedback_entries", [])
            
            # Sort by timestamp (most recent first)
            sorted_entries = sorted(entries, 
                                  key=lambda x: x["timestamp"], 
                                  reverse=True)
            
            return sorted_entries[:limit]
            
        except Exception as e:
            st.error(f"Error loading recent feedback: {e}")
            return []
    
    def export_feedback_csv(self) -> Optional[str]:
        """
        Export feedback to CSV format
        
        Returns:
            CSV content as string or None if error
        """
        try:
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            entries = feedback_data.get("feedback_entries", [])
            
            if not entries:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(entries)
            
            # Convert to CSV
            csv_content = df.to_csv(index=False)
            return csv_content
            
        except Exception as e:
            st.error(f"Error exporting feedback: {e}")
            return None


def render_feedback_widget(feedback_type: str, query: str = "", 
                          response: str = "") -> None:
    """
    Render feedback widget in Streamlit
    
    Args:
        feedback_type: Type of interaction being rated
        query: Original user query
        response: System response
    """
    st.markdown("---")
    st.markdown("**Was this helpful?**")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{feedback_type}_{hash(query[:50])}"):
            feedback_manager = FeedbackManager()
            success = feedback_manager.add_feedback(
                feedback_type=feedback_type,
                rating="thumbs_up",
                query=query,
                response=response
            )
            if success:
                st.success("Thank you for your feedback!")
            else:
                st.error("Error saving feedback")
    
    with col2:
        if st.button("👎", key=f"thumbs_down_{feedback_type}_{hash(query[:50])}"):
            feedback_manager = FeedbackManager()
            success = feedback_manager.add_feedback(
                feedback_type=feedback_type,
                rating="thumbs_down",
                query=query,
                response=response
            )
            if success:
                st.success("Thank you for your feedback!")
                # Option to provide additional comments
                with st.expander("Tell us more (optional)"):
                    comments = st.text_area("How can we improve?", key=f"comments_{feedback_type}_{hash(query[:50])}")
                    if st.button("Submit Comments", key=f"submit_comments_{feedback_type}_{hash(query[:50])}"):
                        feedback_manager.add_feedback(
                            feedback_type=feedback_type,
                            rating="thumbs_down",
                            query=query,
                            response=response,
                            comments=comments
                        )
                        st.success("Comments submitted!")
            else:
                st.error("Error saving feedback")


def display_feedback_dashboard():
    """Display feedback dashboard for administrators"""
    st.subheader("📊 Feedback Dashboard")
    
    feedback_manager = FeedbackManager()
    stats = feedback_manager.get_feedback_stats()
    
    if stats.get("total_feedback", 0) == 0:
        st.info("No feedback received yet.")
        return
    
    # Display overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Feedback", stats["total_feedback"])
    
    with col2:
        st.metric("👍 Positive", stats["thumbs_up"])
    
    with col3:
        st.metric("👎 Negative", stats["thumbs_down"])
    
    with col4:
        st.metric("Satisfaction Rate", f"{stats['satisfaction_rate']:.1f}%")
    
    # Feedback by type
    st.markdown("### Feedback by Feature")
    feedback_by_type = stats.get("feedback_by_type", {})
    
    for feature, counts in feedback_by_type.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{feature.title()}**")
        with col2:
            st.write(f"👍 {counts['thumbs_up']}")
        with col3:
            st.write(f"👎 {counts['thumbs_down']}")
    
    # Recent feedback
    st.markdown("### Recent Feedback")
    recent_feedback = feedback_manager.get_recent_feedback(limit=5)
    
    for feedback in recent_feedback:
        with st.expander(f"{feedback['feedback_type']} - {feedback['rating']} - {feedback['timestamp'][:10]}"):
            st.write(f"**Query:** {feedback.get('query', 'N/A')[:100]}...")
            st.write(f"**Rating:** {feedback['rating']}")
            if feedback.get('comments'):
                st.write(f"**Comments:** {feedback['comments']}")
    
    # Export option
    if st.button("Export Feedback CSV"):
        csv_content = feedback_manager.export_feedback_csv()
        if csv_content:
            st.download_button(
                label="Download CSV",
                data=csv_content,
                file_name=f"feedback_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            ) 