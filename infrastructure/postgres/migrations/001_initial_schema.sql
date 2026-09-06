CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE person_tracks (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('TRACKING', 'LOST', 'ENDED')),
    started_at TIMESTAMPTZ NOT NULL,
    lost_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    CHECK (lost_at IS NULL OR lost_at >= started_at),
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

CREATE TABLE person_profiles (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL CHECK (btrim(name) <> ''),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    CHECK (updated_at >= created_at)
);

CREATE TABLE observation_sessions (
    id UUID PRIMARY KEY,
    person_track_id UUID NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'ENDED')),
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    current_identity_status TEXT NOT NULL
        CHECK (current_identity_status IN ('ANALYZING', 'IDENTIFIED', 'EXTERNAL')),
    current_person_profile_id UUID,
    current_identity_decision_id UUID,
    current_identity_evaluated_at TIMESTAMPTZ NOT NULL,
    CHECK (ended_at IS NULL OR ended_at >= started_at),
    CHECK (
        (current_identity_status = 'IDENTIFIED'
            AND current_person_profile_id IS NOT NULL
            AND current_identity_decision_id IS NOT NULL)
        OR
        (current_identity_status IN ('ANALYZING', 'EXTERNAL')
            AND current_person_profile_id IS NULL)
    )
);

CREATE TABLE face_samples (
    id UUID PRIMARY KEY,
    observation_session_id UUID NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL,
    bbox_left DOUBLE PRECISION NOT NULL,
    bbox_top DOUBLE PRECISION NOT NULL,
    bbox_right DOUBLE PRECISION NOT NULL,
    bbox_bottom DOUBLE PRECISION NOT NULL,
    quality_score DOUBLE PRECISION NOT NULL,
    quality_factors JSONB NOT NULL,
    yaw_degrees DOUBLE PRECISION NOT NULL,
    pitch_degrees DOUBLE PRECISION NOT NULL,
    roll_degrees DOUBLE PRECISION NOT NULL,
    face_crop_storage_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL,
    CHECK (bbox_right > bbox_left),
    CHECK (bbox_bottom > bbox_top)
);

CREATE TABLE face_sample_embeddings (
    face_sample_id UUID PRIMARY KEY,
    embedding_vector vector(512) NOT NULL,
    embedding_model_version TEXT NOT NULL CHECK (btrim(embedding_model_version) <> ''),
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE person_profile_face_templates (
    id UUID PRIMARY KEY,
    person_profile_id UUID NOT NULL,
    source_face_sample_id UUID NOT NULL UNIQUE,
    embedding_vector vector(512) NOT NULL,
    embedding_model_version TEXT NOT NULL CHECK (btrim(embedding_model_version) <> ''),
    enrolled_at TIMESTAMPTZ NOT NULL,
    UNIQUE (id, person_profile_id)
);

CREATE TABLE identity_decisions (
    id UUID PRIMARY KEY,
    face_sample_id UUID NOT NULL UNIQUE,
    candidate_person_profile_id UUID,
    candidate_person_profile_face_template_id UUID,
    similarity DOUBLE PRECISION,
    decided_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (candidate_person_profile_id IS NULL
            AND candidate_person_profile_face_template_id IS NULL
            AND similarity IS NULL)
        OR
        (candidate_person_profile_id IS NOT NULL
            AND candidate_person_profile_face_template_id IS NOT NULL
            AND similarity IS NOT NULL)
    ),
);

CREATE TABLE registration_proposals (
    id UUID PRIMARY KEY,
    observation_session_id UUID NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('PENDING', 'ACCEPTED', 'REJECTED', 'EXPIRED')),
    accepted_name TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    responded_at TIMESTAMPTZ,
    CHECK (expires_at > created_at),
    CHECK (
        (status = 'PENDING' AND accepted_name IS NULL AND responded_at IS NULL)
        OR (status = 'ACCEPTED' AND accepted_name IS NOT NULL
            AND btrim(accepted_name) <> '' AND responded_at IS NOT NULL)
        OR (status = 'REJECTED' AND accepted_name IS NULL AND responded_at IS NOT NULL)
        OR (status = 'EXPIRED' AND accepted_name IS NULL AND responded_at IS NULL)
    )
);

CREATE TABLE registration_proposal_face_samples (
    registration_proposal_id UUID NOT NULL,
    face_sample_id UUID NOT NULL,
    selection_order SMALLINT NOT NULL CHECK (selection_order > 0),
    PRIMARY KEY (registration_proposal_id, face_sample_id),
    UNIQUE (registration_proposal_id, selection_order)
);

CREATE TABLE registration_handlings (
    registration_proposal_id UUID PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('REGISTERED', 'FAILED')),
    person_profile_id UUID,
    failure_reason TEXT,
    completed_at TIMESTAMPTZ NOT NULL,
    CHECK (
        (status = 'REGISTERED' AND person_profile_id IS NOT NULL AND failure_reason IS NULL)
        OR (status = 'FAILED' AND person_profile_id IS NULL AND failure_reason IS NOT NULL
            AND btrim(failure_reason) <> '')
    )
);

CREATE INDEX person_profile_face_templates_profile_idx
    ON person_profile_face_templates (person_profile_id);
CREATE INDEX observation_sessions_started_at_idx ON observation_sessions (started_at);
CREATE INDEX face_samples_session_captured_at_idx
    ON face_samples (observation_session_id, captured_at);
CREATE INDEX registration_proposals_status_expires_at_idx
    ON registration_proposals (status, expires_at);
CREATE INDEX registration_handlings_person_profile_id_idx
    ON registration_handlings (person_profile_id);
