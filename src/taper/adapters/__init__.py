"""Readers for formats this engine did not invent.

The rest of the repository reconciles data the generator produced. Everything
in here reads a shape somebody else defined, which is where the interesting
mistakes live - subunits read as units, epochs read as dates, a boolean that
decides whether money exists.
"""
