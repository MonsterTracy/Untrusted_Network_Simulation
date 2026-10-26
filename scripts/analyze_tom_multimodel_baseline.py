"""Rebuild research reports from a complete sealed ToM multimodel experiment."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from werewolf.tom_llm.baseline_cli import main
if __name__=='__main__':main(analysis=True)
