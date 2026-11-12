from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import String, Integer, Text, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.base import Base


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    contract_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("contracts.id", ondelete="CASCADE"), index=True, nullable=False
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    contract: Mapped["Contract"] = relationship("Contract", back_populates="analysis_runs")
    clauses: Mapped[List["ExtractedClauseModel"]] = relationship(
        "ExtractedClauseModel", back_populates="analysis_run", cascade="all, delete-orphan"
    )
    risk_assessments: Mapped[List["RiskAssessmentModel"]] = relationship(
        "RiskAssessmentModel", back_populates="analysis_run", cascade="all, delete-orphan"
    )

