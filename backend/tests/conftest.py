import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app

# In-memory SQLite engine for fast, isolated test runs
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden database dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_contract_text():
    return """MASTER SERVICES AGREEMENT

1. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and continue for one (1) year. Either party may terminate this Agreement at any time without cause upon 30 days' prior written notice to the other party.

2. AUTOMATIC RENEWAL
This Agreement shall automatically renew for successive terms of twelve (12) months unless either party provides written notice of non-renewal at least sixty (60) days prior to the expiration of the current term.

3. LIMITATION OF LIABILITY
In no event shall either party's aggregate liability arising out of or related to this Agreement exceed the total amounts paid or payable by Customer hereunder in the twelve (12) months preceding the incident. Neither party will be liable for any indirect, special, incidental, or consequential damages.

4. ASSIGNMENT
Neither party may assign, delegate, or transfer any of its rights or obligations under this Agreement without the prior written consent of the other party, except in connection with a merger or sale of substantially all assets.

5. GOVERNING LAW AND JURISDICTION
This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to conflict of law principles. Any legal action shall be brought exclusively in the courts located in New Castle County, Delaware.

6. AUDIT RIGHTS
Customer shall have the right to audit and inspect the books, records, and facilities of Vendor upon reasonable notice during normal business hours to ensure compliance with this Agreement.

7. INTELLECTUAL PROPERTY
All right, title, and interest in and to all deliverables, inventions, and work product created hereunder shall belong exclusively to Customer as a work made for hire.

8. LICENSE GRANT
Vendor hereby grants to Customer a worldwide, non-exclusive, royalty-free license to use the Software and documentation solely for internal business operations.

9. NON-COMPETITION
During the term of this Agreement and for a period of one (1) year following termination, Contractor shall not directly or indirectly engage in any competitive business with Company within the territory.
"""


@pytest.fixture
def sample_docx_bytes(sample_contract_text):
    """Generates a real .docx in-memory bytes object."""
    import docx
    doc = docx.Document()
    for paragraph in sample_contract_text.split("\n\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


@pytest.fixture
def sample_pdf_bytes(sample_contract_text):
    """Generates a real .pdf in-memory bytes object."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), sample_contract_text, fontsize=10)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

