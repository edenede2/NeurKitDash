"""
Enhanced logging system for subject data loading process
Provides detailed logging with different levels and structured output
"""

import logging
import os
import datetime
import json
from typing import Dict, List, Any, Optional
from uuid import uuid4

class LoadingLogger:
    """Enhanced logger for the data loading process"""

    def __init__(self, project: str, log_dir: str = None):
        """
        Initialize the loading logger

        Args:
            project: Project name
            log_dir: Directory to store log files (default: logs/)
        """
        self.project = project
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), "logs")
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]

        # Create logs directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Setup main logger
        self.logger = self._setup_main_logger()

        # Session tracking
        self.session_stats = {
            'project': project,
            'session_id': self.session_id,
            'start_time': datetime.datetime.now().isoformat(),
            'subjects_processed': [],
            'subjects_failed': [],
            'subjects_skipped': [],
            'processing_stats': {},
            'errors': [],
            'warnings': []
        }

        # Subject-specific tracking
        self.current_subject = None
        self.subject_stats = {}

    def _setup_main_logger(self) -> logging.Logger:
        """Setup the main logger with file and console handlers"""
        logger_name = f"loading_{self.project}_{self.session_id}"
        logger = logging.Logger(logger_name)
        logger.setLevel(logging.DEBUG)

        # File handler for detailed logs
        log_file = os.path.join(self.log_dir, f"{self.project}_loading_{self.session_id}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Detailed formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Simple formatter for console
        simple_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )

        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Log session start
        logger.info(f"{'='*80}")
        logger.info(f"LOADING SESSION STARTED: {self.project}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"{'='*80}")

        return logger

    def close(self):
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

    def start_subject_processing(self, subject_id: str, subject_path: str = None):
        """Start processing a new subject"""
        self.current_subject = subject_id
        self.subject_stats[subject_id] = {
            'subject_id': subject_id,
            'subject_path': subject_path,
            'start_time': datetime.datetime.now().isoformat(),
            'status': 'processing',
            'signals_detected': [],
            'files_found': {},
            'processing_steps': [],
            'errors': [],
            'warnings': [],
            'data_quality': {},
            'processing_time': None
        }

        self.logger.info(f"\n{'-'*60}")
        self.logger.info(f"STARTING SUBJECT: {subject_id}")
        self.logger.info(f"Subject path: {subject_path}")
        self.logger.info(f"{'-'*60}")

    def log_file_discovery(self, file_type: str, file_path: str, file_size: int = None):
        """Log discovered files for current subject"""
        if self.current_subject:
            self.subject_stats[self.current_subject]['files_found'][file_type] = {
                'path': file_path,
                'size_bytes': file_size,
                'exists': os.path.exists(file_path)
            }

        size_str = f" ({file_size:,} bytes)" if file_size else ""
        self.logger.info(f"📁 Found {file_type}: {os.path.basename(file_path)}{size_str}")

    def log_signal_detection(self, signal_type: str, detected: bool, channel_info: Dict = None):
        """Log signal type detection"""
        if self.current_subject:
            signal_info = {
                'type': signal_type,
                'detected': detected,
                'channel_info': channel_info or {}
            }
            self.subject_stats[self.current_subject]['signals_detected'].append(signal_info)

        status = "✅ DETECTED" if detected else "❌ NOT FOUND"
        self.logger.info(f"🔍 {signal_type.upper()} signal: {status}")
        if channel_info:
            self.logger.debug(f"   Channel info: {channel_info}")

    def log_processing_step(self, step_name: str, success: bool, details: Dict = None):
        """Log individual processing steps"""
        if self.current_subject:
            step_info = {
                'step': step_name,
                'success': success,
                'timestamp': datetime.datetime.now().isoformat(),
                'details': details or {}
            }
            self.subject_stats[self.current_subject]['processing_steps'].append(step_info)

        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.logger.info(f"🔧 {step_name}: {status}")
        if details:
            self.logger.debug(f"   Details: {details}")

    def log_data_quality(self, quality_metrics: Dict):
        """Log data quality metrics"""
        if self.current_subject:
            self.subject_stats[self.current_subject]['data_quality'].update(quality_metrics)

        self.logger.info(f"📊 Data Quality Metrics:")
        for metric, value in quality_metrics.items():
            self.logger.info(f"   {metric}: {value}")

    def log_warning(self, message: str, details: Dict = None):
        """Log a warning"""
        self.logger.warning(message)
        warning_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'subject': self.current_subject,
            'message': message,
            'details': details or {}
        }
        self.session_stats['warnings'].append(warning_entry)

        if self.current_subject and self.current_subject in self.subject_stats:
            self.subject_stats[self.current_subject]['warnings'].append(warning_entry)

    def log_error(self, message: str, exception: Exception = None, details: Dict = None):
        """Log an error"""
        error_msg = f"{message}"
        if exception:
            error_msg += f" | Exception: {str(exception)}"

        self.logger.error(error_msg, exc_info=(type(exception), exception, exception.__traceback__) if exception else None)

        error_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'subject': self.current_subject,
            'message': message,
            'exception': str(exception) if exception else None,
            'exception_type': type(exception).__name__ if exception else None,
            'details': details or {}
        }
        self.session_stats['errors'].append(error_entry)

        if self.current_subject and self.current_subject in self.subject_stats:
            self.subject_stats[self.current_subject]['errors'].append(error_entry)

    def finish_subject_processing(self, status: str, final_message: str = None, *, subject_id=None):
        """Finish processing current subject"""
        subject_id = subject_id or self.current_subject
        if not subject_id:
            return
        start_time = datetime.datetime.fromisoformat(self.subject_stats[subject_id]['start_time'])
        processing_time = (datetime.datetime.now() - start_time).total_seconds()

        self.subject_stats[subject_id].update({
            'status': status,
            'end_time': datetime.datetime.now().isoformat(),
            'processing_time': processing_time,
            'final_message': final_message
        })

        # Update session stats
        if status == 'success':
            self.session_stats['subjects_processed'].append(subject_id)
        elif status == 'failed':
            self.session_stats['subjects_failed'].append(subject_id)
        elif status == 'skipped':
            self.session_stats['subjects_skipped'].append(subject_id)

        # Log completion
        status_emoji = {
            'success': '✅',
            'failed': '❌',
            'skipped': '⏭️'
        }.get(status, '❔')

        self.logger.info(f"{status_emoji} FINISHED SUBJECT: {subject_id} ({status.upper()})")
        self.logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
        if final_message:
            self.logger.info(f"💬 Final message: {final_message}")
        self.logger.info(f"{'-'*60}")

        self.current_subject = None

    def generate_session_summary(self) -> Dict:
        """Generate comprehensive session summary"""
        end_time = datetime.datetime.now()
        start_time = datetime.datetime.fromisoformat(self.session_stats['start_time'])
        total_time = (end_time - start_time).total_seconds()

        summary = {
            **self.session_stats,
            'end_time': end_time.isoformat(),
            'total_time_seconds': total_time,
            'total_subjects_found': len(self.subject_stats),
            'success_count': len(self.session_stats['subjects_processed']),
            'failed_count': len(self.session_stats['subjects_failed']),
            'skipped_count': len(self.session_stats['subjects_skipped']),
            'error_count': len(self.session_stats['errors']),
            'warning_count': len(self.session_stats['warnings']),
            'subject_details': self.subject_stats
        }

        return summary

    def save_session_report(self) -> str:
        """Save comprehensive session report"""
        summary = self.generate_session_summary()

        # Save JSON report
        report_file = os.path.join(self.log_dir, f"{self.project}_loading_report_{self.session_id}.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Save human-readable summary
        summary_file = os.path.join(self.log_dir, f"{self.project}_loading_summary_{self.session_id}.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._format_summary_text(summary))

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"SESSION COMPLETED")
        self.logger.info(f"📊 Summary: {summary['success_count']} success, {summary['failed_count']} failed, {summary['skipped_count']} skipped")
        self.logger.info(f"⏱️  Total time: {summary['total_time_seconds']:.2f} seconds")
        self.logger.info(f"📄 Report saved: {report_file}")
        self.logger.info(f"📄 Summary saved: {summary_file}")
        self.logger.info(f"{'='*80}")

        return report_file

    def _format_summary_text(self, summary: Dict) -> str:
        """Format summary as human-readable text"""
        text = []
        text.append(f"LOADING SESSION SUMMARY - {summary['project'].upper()}")
        text.append("=" * 80)
        text.append(f"Session ID: {summary['session_id']}")
        text.append(f"Start Time: {summary['start_time']}")
        text.append(f"End Time: {summary['end_time']}")
        text.append(f"Total Duration: {summary['total_time_seconds']:.2f} seconds")
        text.append("")

        text.append("PROCESSING RESULTS:")
        text.append(f"  Total subjects found: {summary['total_subjects_found']}")
        text.append(f"  ✅ Successfully processed: {summary['success_count']}")
        text.append(f"  ❌ Failed to process: {summary['failed_count']}")
        text.append(f"  ⏭️  Skipped: {summary['skipped_count']}")
        text.append(f"  ⚠️  Warnings: {summary['warning_count']}")
        text.append(f"  🚨 Errors: {summary['error_count']}")
        text.append("")

        if summary['subjects_processed']:
            text.append("SUCCESSFULLY PROCESSED SUBJECTS:")
            for subject in summary['subjects_processed']:
                text.append(f"  ✅ {subject}")
            text.append("")

        if summary['subjects_failed']:
            text.append("FAILED SUBJECTS:")
            for subject in summary['subjects_failed']:
                subject_info = summary['subject_details'].get(subject, {})
                errors = subject_info.get('errors', [])
                error_msg = errors[-1]['message'] if errors else "Unknown error"
                text.append(f"  ❌ {subject}: {error_msg}")
            text.append("")

        if summary['subjects_skipped']:
            text.append("SKIPPED SUBJECTS:")
            for subject in summary['subjects_skipped']:
                text.append(f"  ⏭️  {subject}")
            text.append("")

        # Subject details
        text.append("DETAILED SUBJECT INFORMATION:")
        text.append("-" * 80)
        for subject_id, details in summary['subject_details'].items():
            text.append(f"\nSUBJECT: {subject_id}")
            text.append(f"  Status: {details['status']}")
            text.append(f"  Processing time: {details.get('processing_time', 0):.2f}s")

            signals = details.get('signals_detected', [])
            if signals:
                text.append(f"  Signals detected:")
                for signal in signals:
                    status = "✅" if signal['detected'] else "❌"
                    text.append(f"    {status} {signal['type'].upper()}")

            files = details.get('files_found', {})
            if files:
                text.append(f"  Files found:")
                for file_type, file_info in files.items():
                    status = "✅" if file_info['exists'] else "❌"
                    size = f" ({file_info['size_bytes']:,} bytes)" if file_info.get('size_bytes') else ""
                    text.append(f"    {status} {file_type}: {os.path.basename(file_info['path'])}{size}")

            quality = details.get('data_quality', {})
            if quality:
                text.append(f"  Data quality:")
                for metric, value in quality.items():
                    text.append(f"    {metric}: {value}")

            errors = details.get('errors', [])
            if errors:
                text.append(f"  Errors ({len(errors)}):")
                for error in errors[-3:]:  # Show last 3 errors
                    text.append(f"    - {error['message']}")

            warnings = details.get('warnings', [])
            if warnings:
                text.append(f"  Warnings ({len(warnings)}):")
                for warning in warnings[-3:]:  # Show last 3 warnings
                    text.append(f"    - {warning['message']}")

        return "\n".join(text)

    def log_schema_mismatch(self, schema_type: str, existing_cols: set, new_cols: set, resolution: str = None):
        """Log schema mismatch issues and their resolution"""
        missing_in_existing = new_cols - existing_cols
        missing_in_new = existing_cols - new_cols

        details = {
            'existing_columns': sorted(list(existing_cols)),
            'new_columns': sorted(list(new_cols)),
            'missing_in_existing': sorted(list(missing_in_existing)),
            'missing_in_new': sorted(list(missing_in_new)),
            'resolution': resolution
        }

        self.log_warning(f"Schema mismatch detected in {schema_type}", details)

        if missing_in_existing:
            self.logger.warning(f"   Missing in existing: {', '.join(missing_in_existing)}")
        if missing_in_new:
            self.logger.warning(f"   Missing in new: {', '.join(missing_in_new)}")
        if resolution:
            self.logger.info(f"   Resolution: {resolution}")

    def get_problematic_subjects(self) -> Dict[str, List[str]]:
        """Get list of subjects that had issues"""
        problematic = {
            'failed': self.session_stats['subjects_failed'].copy(),
            'with_errors': [],
            'with_warnings': [],
            'poor_quality': []
        }

        for subject_id, details in self.subject_stats.items():
            if details.get('errors') and subject_id not in problematic['failed']:
                problematic['with_errors'].append(subject_id)

            if details.get('warnings'):
                problematic['with_warnings'].append(subject_id)

            # Check for poor data quality indicators
            quality = details.get('data_quality', {})
            if any(
                key.endswith('_quality') and isinstance(value, (int, float)) and value < 0.5
                for key, value in quality.items()
            ):
                problematic['poor_quality'].append(subject_id)

        return problematic

    def log_debug(self, message: str, details: Dict = None):
        """Log debug information for detailed troubleshooting"""
        debug_msg = f"[DEBUG] {message}"
        if details:
            debug_msg += f" | Details: {details}"

        self.logger.debug(debug_msg)

        # Optional: store debug messages if needed for analysis
        # (uncomment if you want to track debug messages in session stats)
        # debug_entry = {
        #     'timestamp': datetime.datetime.now().isoformat(),
        #     'subject': self.current_subject,
        #     'message': message,
        #     'details': details or {}
        # }
        # if 'debug_messages' not in self.session_stats:
        #     self.session_stats['debug_messages'] = []
        # self.session_stats['debug_messages'].append(debug_entry)
