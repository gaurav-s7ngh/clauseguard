from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class RiskAssessmentModel(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    clause_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("extracted_clauses.id", ondelete="CASCADE"), unique=True, index=True, nullable=False
    )
    analysis_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    risk_level: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    risk_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    risk_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    plain_english_explanation: Mapped[str] = mapped_column(Text, nullable=False)
    suggested_modification: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    clause: Mapped["ExtractedClauseModel"] = relationship("ExtractedClauseModel", back_populates="risk_assessment")
    analysis_run: Mapped["AnalysisRun"] = relationship("AnalysisRun", back_populates="risk_assessments")

