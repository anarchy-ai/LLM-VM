# Generated by Django 4.2.6 on 2023-10-28 22:49

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='LanguageModel',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model', models.CharField(max_length=200)),
                ('model_size', models.CharField(max_length=200)),
                ('prompt', models.TextField()),
            ],
            options={
                'ordering': ['id'],
            },
        ),
    ]
