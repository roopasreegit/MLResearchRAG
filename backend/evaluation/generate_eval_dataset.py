from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

from backend.retrieval.retrieve import collection

all_docs=collection.get()
documents=all_docs["documents"]

syn = Synthesizer(
    styling_config=StylingConfig(
        scenario="GenAI Research paper question answering"
    )
)

goldens = syn.generate_goldens_from_docs(
    document_chunks=documents,
    max_goldens_per_document=2
)

for golden in goldens:
    print("\nQUESTION:")
    print(golden.input)

    print("\nEXPECTED ANSWER:")
    print(golden.expected_output)