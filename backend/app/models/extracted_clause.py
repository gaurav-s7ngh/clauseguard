from typing import Optional
from sqlalchemy import String, Integer, Float, Text, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class ExtractedClauseModel(Base):
    __tablename__ = "extracted_clauses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    analysis_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    contract_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("contracts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    category: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    clause_text: Mapped[str] = mapped_column(Text, nullable=False)
    start_char: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    surrounding_context: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_status: Mapped[str] = mapped_column(String(32), default="extracted", nullable=False)

    # Relationships
    contract: Mapped["Contract"] = relationship("Contract", back_populates="extracted_clauses")
    analysis_run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="clauses")
    risk_assessment: Mapped[Optional["RiskAssessmentModel"]] = relationship(
        "RiskAssessmentModel", back_populates="clause", uselist=False, cascade="all, delete-orphan"
    )

