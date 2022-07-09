"""empty message

Revision ID: 92bcaa9e5597
Revises: 
Create Date: 2019-05-09 21:43:51.799368

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '92bcaa9e5597'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('t_test')
    op.drop_table('channels')
    op.drop_table('integrations')
    op.drop_index('ix_developers_dev_key', table_name='developers')
    op.drop_index('ix_developers_username', table_name='developers')
    op.drop_table('developers')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('developers',
    sa.Column('id', sa.INTEGER(), server_default=sa.text("nextval('developers_id_seq'::regclass)"), autoincrement=True, nullable=False),
    sa.Column('dev_key', sa.VARCHAR(length=40), autoincrement=False, nullable=True),
    sa.Column('platform', sa.VARCHAR(length=50), autoincrement=False, nullable=True),
    sa.Column('platform_id', sa.VARCHAR(length=40), autoincrement=False, nullable=True),
    sa.Column('username', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='developers_pkey'),
    sa.UniqueConstraint('platform_id', name='developers_platform_id_key'),
    postgresql_ignore_search_path=False
    )
    op.create_index('ix_developers_username', 'developers', ['username'], unique=False)
    op.create_index('ix_developers_dev_key', 'developers', ['dev_key'], unique=True)
    op.create_table('integrations',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('integration_id', sa.VARCHAR(length=40), autoincrement=False, nullable=True),
    sa.Column('name', sa.VARCHAR(length=100), autoincrement=False, nullable=True),
    sa.Column('description', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.Column('icon', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.Column('channel', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.Column('token', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.Column('developer_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['developer_id'], ['developers.id'], name='integrations_developer_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='integrations_pkey'),
    sa.UniqueConstraint('integration_id', name='integrations_integration_id_key')
    )
    op.create_table('channels',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('developer_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('channel', sa.VARCHAR(length=150), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['developer_id'], ['developers.id'], name='channels_developer_id_fkey'),
    sa.PrimaryKeyConstraint('id', name='channels_pkey')
    )
    op.create_table('t_test',
    sa.Column('id', postgresql.INT4RANGE(), autoincrement=False, nullable=False),
    sa.Column('age', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('salary', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='t_test_pkey')
    )
    # ### end Alembic commands ###
