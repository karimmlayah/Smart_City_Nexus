"""
AI Agents for Traffic Violation Analysis
Uses OpenAI API with function calling for intelligent violation processing
"""
import os
import json
from datetime import datetime
from typing import Dict, List, Any
import requests


class ViolationAnalysisAgent:
    """AI agent that analyzes violation data and generates reports"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def analyze_violation(self, violation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a traffic violation and generate a detailed report
        """
        prompt = f"""You are a traffic violation analysis expert. Analyze this violation data and provide a comprehensive report.

Violation Data:
- Car ID: {violation_data.get('car_id')}
- Traffic Light State: {violation_data.get('traffic_light')}
- Crossed Stop Line: {violation_data.get('crossed_stop_line')}
- Plate Images Count: {violation_data.get('plate_images_count', 0)}
- Detection Confidence: {violation_data.get('confidence', 0):.2%}
- Timestamp: {violation_data.get('timestamp')}
- Frame Number: {violation_data.get('frame_number')}

Provide:
1. Violation confirmation (yes/no with reasoning)
2. Severity level (minor/moderate/severe)
3. Evidence quality assessment
4. Recommended action
5. Additional observations

Be professional and precise."""

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a traffic violation analysis expert. Provide clear, professional assessments."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "analysis": analysis,
                    "model": "gpt-4o-mini"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class ViolationSummaryAgent:
    """AI agent that generates executive summaries of multiple violations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_summary(self, violations: List[Dict[str, Any]], video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate an executive summary of all violations in a video
        """
        prompt = f"""You are a traffic monitoring system analyst. Generate an executive summary report.

Video Analysis Summary:
- Total Violations: {len(violations)}
- Video Duration: {video_metadata.get('duration', 'N/A')}
- Frames Processed: {video_metadata.get('frames_processed', 0)}
- Detection Date: {video_metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}

Violations:
{json.dumps(violations, indent=2)}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Statistics
3. Patterns Observed
4. Risk Assessment
5. Recommendations for Traffic Authority

Be concise and professional."""

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a traffic monitoring analyst. Provide executive-level insights."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.4,
                    "max_tokens": 600
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "summary": summary,
                    "model": "gpt-4o-mini"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class RecommendationAgent:
    """AI agent that provides actionable recommendations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def get_recommendations(self, violation_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on violation patterns
        """
        prompt = f"""You are a traffic safety consultant. Based on these violation statistics, provide actionable recommendations.

Statistics:
- Total Violations: {violation_stats.get('total_violations', 0)}
- Violation Rate: {violation_stats.get('violation_rate', 0):.2%}
- Peak Violation Time: {violation_stats.get('peak_time', 'N/A')}
- Most Common Vehicle Type: {violation_stats.get('common_vehicle', 'N/A')}
- Average Confidence: {violation_stats.get('avg_confidence', 0):.2%}

Provide:
1. Infrastructure Improvements (2-3 suggestions)
2. Enforcement Strategies (2-3 suggestions)
3. Public Awareness Campaigns (2-3 suggestions)
4. Technology Enhancements (2-3 suggestions)
5. Priority Actions (top 3)

Be specific and actionable."""

        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a traffic safety consultant. Provide practical, actionable recommendations."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 700
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                recommendations = result["choices"][0]["message"]["content"]
                return {
                    "success": True,
                    "recommendations": recommendations,
                    "model": "gpt-4o-mini"
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
