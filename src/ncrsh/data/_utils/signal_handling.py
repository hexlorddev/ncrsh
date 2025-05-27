"""Signal handling utilities for DataLoader workers."""
import os
import signal
import sys
import threading
from typing import Any, Callable, Optional

# Global variable to track if we're in a worker process
_worker_process = False

# Original signal handlers
_original_sigint_handler = signal.getsignal(signal.SIGINT)
_original_sigterm_handler = signal.getsignal(signal.SIGTERM)

def _set_worker_signal_handlers() -> None:
    """Set up signal handlers for worker processes.
    
    This ensures that worker processes exit cleanly when interrupted.
    """
    global _worker_process
    _worker_process = True
    
    # Reset signal handlers to default in worker processes
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # Ignore SIGPIPE
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

def _disable_worker_signal_handlers() -> None:
    """Disable signal handlers in the main process.
    
    This is used when the main process needs to handle signals itself.
    """
    global _worker_process
    _worker_process = False
    
    # Restore original signal handlers
    signal.signal(signal.SIGINT, _original_sigint_handler)
    signal.signal(signal.SIGTERM, _original_sigterm_handler)

def _is_main_process() -> bool:
    """Check if the current process is the main process."""
    return not _worker_process and threading.current_thread() is threading.main_thread()

def _is_fork() -> bool:
    """Check if the current process is a forked process."""
    return os.getpid() != os.getppid()

def _set_SIGCHLD_handler() -> None:
    """Set up SIGCHLD handler to clean up zombie processes."""
    if sys.platform == 'win32':
        return
        
    # Only set the handler in the main process
    if not _is_main_process():
        return
        
    def handler(signum, frame):
        # Reap zombie processes
        try:
            while True:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid == 0:  # No more zombies
                    break
        except (OSError, InterruptedError):
            pass
    
    # Set the handler
    signal.signal(signal.SIGCHLD, handler)
